import json
import logging
import os
import random
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from dltranz.data_load import ConvertingTrxDataset, DropoutTrxDataset, read_data_gen, AllTimeShuffleMLDataset
from dltranz.experiment import update_model_stats
from dltranz.metric_learn.dataset import SplittingDataset, split_strategy
from dltranz.metric_learn.dataset import TargetEnumeratorDataset, collate_splitted_rows
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.metric import BatchRecallTop
from dltranz.metric_learn.ml_models import ml_model_by_type
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model, CheckpointHandler
from dltranz.util import init_logger, get_conf, switch_reproducibility_on

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def prepare_embeddings(seq, conf, is_train):
    min_seq_len = conf['dataset'].get('min_seq_len', 1)
    embeddings = list(conf['params.trx_encoder.embeddings'].keys())

    feature_keys = embeddings + list(conf['params.trx_encoder.numeric_values'].keys())

    for rec in seq:
        seq_len = len(rec['event_time'])
        if is_train and seq_len < min_seq_len:
            continue

        if 'feature_arrays' in rec:
            feature_arrays = rec['feature_arrays']
            feature_arrays = {k: v for k, v in feature_arrays.items() if k in feature_keys}
        else:
            feature_arrays = {k: v for k, v in rec.items() if k in feature_keys}

        # TODO: datetime processing. Take date-time features

        # shift embeddings to 1, 0 is padding value
        feature_arrays = {k: v + (1 if k in embeddings else 0) for k, v in feature_arrays.items()}

        # clip embeddings dictionary by max value
        for e_name, e_params in conf['params.trx_encoder.embeddings'].items():
            feature_arrays[e_name] = feature_arrays[e_name].clip(0, e_params['in'] - 1)

        feature_arrays['event_time'] = rec['event_time']

        rec['feature_arrays'] = feature_arrays
        yield rec


def shuffle_client_list_reproducible(conf, data):
    if conf['dataset.client_list_shuffle_seed'] != 0:
        dataset_col_id = conf['dataset'].get('col_id', 'client_id')
        data = sorted(data, key=lambda x: x.get(dataset_col_id, x.get('customer_id', x.get('installation_id'))))
        random.Random(conf['dataset.client_list_shuffle_seed']).shuffle(data)
    return data


def _seq_len_stats(data):
    lengths = [len(rec['event_time']) for rec in data]
    lengths = np.array(lengths)
    return {
        'n': int(len(lengths)),
        'min': int(lengths.min()),
        'max': int(lengths.max()),
        'mean': float(lengths.mean()),
        'median': float(np.median(lengths)),
        'p25': float(np.percentile(lengths, 25)),
        'p75': float(np.percentile(lengths, 75)),
        'p95': float(np.percentile(lengths, 95)),
    }


def _feature_stats(data, feature_keys):
    stats = {}
    for key in feature_keys:
        arrays = [rec['feature_arrays'][key] for rec in data if key in rec.get('feature_arrays', {})]
        if not arrays:
            continue
        values = np.concatenate(arrays)
        stats[key] = {
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'std': float(values.std()),
        }
    return stats


def _to_int_id(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _spot_check_sequences(data, col_id, feature_keys, n_clients=3):
    spot = []
    for rec in data[:n_clients]:
        client_id = rec.get(col_id, rec.get('customer_id', rec.get('installation_id')))
        entry = {
            'client_id': _to_int_id(client_id),
            'seq_len': int(len(rec['event_time'])),
            'event_time': rec['event_time'].tolist(),
        }
        for key in feature_keys:
            if key in rec.get('feature_arrays', {}):
                entry[key] = rec['feature_arrays'][key].tolist()
        spot.append(entry)
    return spot


def _batch_feature_stats(values_2d, seq_lens):
    """Per-feature stats over (B, T) tensor: padded (all entries) and valid-only.

    Padding values are 0 (from torch.nn.utils.rnn.pad_sequence). Valid stats
    iterate explicitly to avoid masking ambiguity when a real value can equal 0.
    """
    padded = values_2d.detach().cpu().numpy().astype(float)
    padded_stats = {
        'min': float(padded.min()),
        'max': float(padded.max()),
        'mean': float(padded.mean()),
        'std': float(padded.std()),
    }
    valid_chunks = []
    for i, sl in enumerate(seq_lens.tolist()):
        valid_chunks.append(padded[i, :sl])
    valid = np.concatenate(valid_chunks) if valid_chunks else padded.reshape(-1)
    valid_stats = {
        'min': float(valid.min()),
        'max': float(valid.max()),
        'mean': float(valid.mean()),
        'std': float(valid.std()),
    }
    return padded_stats, valid_stats


def _batch_spot_check(payload, seq_lens, targets, feature_keys, n=3, head=20):
    spot = []
    sl_list = seq_lens.tolist()
    target_list = targets.tolist() if targets is not None else [None] * len(sl_list)
    for i in range(min(n, len(sl_list))):
        sl = int(sl_list[i])
        entry = {
            'slice_idx': int(i),
            'enumerated_target': int(target_list[i]) if target_list[i] is not None else None,
            'length': sl,
        }
        for key in feature_keys:
            if key not in payload:
                continue
            arr = payload[key][i, :sl].detach().cpu().numpy().tolist()
            entry[key] = arr[:head]
        spot.append(entry)
    return spot


def save_first_batch_snapshot(train_loader, conf):
    """Snapshot-2: capture the first batch as it would be seen by training.

    Saves to data_snapshot_batch.json next to the existing data_snapshot.json.

    RNG state is saved before consuming the batch and restored after, so that
    `fit_model` sees the SAME first batch regardless of whether this snapshot
    was taken or not. Restored state covers: python `random`, `numpy`, `torch`
    (CPU + CUDA), and the DataLoader's `torch.Generator`.
    """
    import torch as _torch  # local alias to avoid shadowing the module-level import

    stats_path = conf['stats.path']
    snapshot_path = os.path.join(os.path.dirname(stats_path), 'data_snapshot_batch.json')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    embeddings = list(conf['params.trx_encoder.embeddings'].keys())
    numerics = list(conf['params.trx_encoder.numeric_values'].keys())
    feature_keys = embeddings + numerics + ['event_time']

    # --- save RNG state ---
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = _torch.get_rng_state()
    cuda_states = _torch.cuda.get_rng_state_all() if _torch.cuda.is_available() else None
    loader_gen_state = (
        train_loader.generator.get_state() if train_loader.generator is not None else None
    )

    seed = conf.get('common_seed', 42)

    # Size of the DataLoader's dataset after prepare_embeddings filtering (min_seq_len).
    # Differs from n_train in snapshot-1 if some clients are filtered out before the split.
    try:
        train_dataset_size = len(train_loader.dataset)
    except Exception:
        train_dataset_size = None

    # What the DataLoader's first iter actually picks: PyTorch's _BaseDataLoaderIter
    # advances the generator by ONE random int (`_base_seed`) before RandomSampler
    # calls randperm. Without this matching advance, expected_randperm_first5 would
    # differ from target.tolist()[0] even though both use seed 42.
    if train_dataset_size is not None:
        _g = _torch.Generator()
        _g.manual_seed(seed)
        _torch.empty((), dtype=_torch.int64).random_(generator=_g)
        _perm_first5 = _torch.randperm(train_dataset_size, generator=_g)[:5].tolist()
    else:
        _perm_first5 = None

    # --- consume first batch ---
    padded_batch, target = next(iter(train_loader))
    payload = padded_batch.payload  # dict[name -> Tensor (B, T)]
    seq_lens = padded_batch.seq_lens  # IntTensor (B,)
    bs = int(seq_lens.shape[0])

    # --- look up raw client_ids from the underlying train_data list ---
    # `target` carries dataset positions (TargetEnumeratorDataset returns (x, idx)),
    # which are positions in SplittingDataset.base_dataset == train_data list.
    col_id = conf['dataset'].get('col_id', 'client_id')
    ds = train_loader.dataset
    chain_types = [type(ds).__name__]
    while hasattr(ds, 'core_dataset'):
        ds = ds.core_dataset
        chain_types.append(type(ds).__name__)
    while hasattr(ds, 'delegate'):
        ds = ds.delegate
        chain_types.append(type(ds).__name__)
    while hasattr(ds, 'base_dataset'):
        ds = ds.base_dataset
        chain_types.append(type(ds).__name__)

    logger.info(f'COLES_PROBE [save_first_batch_snapshot] target.dtype={target.dtype} '
                f'target[:5]={[int(x) for x in target.tolist()[:5]]} '
                f'len(target)={len(target)} bs={bs}')
    logger.info(f'COLES_PROBE [save_first_batch_snapshot] expected_randperm={_perm_first5} '
                f'len(train_loader.dataset)={train_dataset_size}')
    logger.info(f'COLES_PROBE [save_first_batch_snapshot] unwrap_chain={chain_types} '
                f'final_type={type(ds).__name__} '
                f'final_id={id(ds)} '
                f'is_list={isinstance(ds, list)} len={len(ds) if hasattr(ds, "__len__") else "?"}')
    if isinstance(ds, list) and len(ds) > 31992:
        logger.info(f'COLES_PROBE [save_first_batch_snapshot] '
                    f'ds[31992][{col_id}]={ds[31992].get(col_id)!r} '
                    f'ds[31992][event_time][:3]={list(ds[31992].get("event_time", []))[:3]}')
    if len(target) > 0:
        first_t = int(target.tolist()[0])
        logger.info(f'COLES_PROBE [save_first_batch_snapshot] target[0]={first_t} '
                    f'ds[target[0]][{col_id}]='
                    f'{ds[first_t].get(col_id) if isinstance(ds, list) and 0 <= first_t < len(ds) else "OOB"!r}')
    raw_client_ids = []
    raw_client_ids_ok = True
    try:
        for t in target.tolist():
            rec = ds[int(t)]
            cid = rec.get(col_id, int(t)) if isinstance(rec, dict) else int(t)
            try:
                raw_client_ids.append(int(cid))
            except (TypeError, ValueError):
                raw_client_ids.append(str(cid))
    except (TypeError, KeyError, IndexError, AttributeError):
        raw_client_ids_ok = False
        raw_client_ids = None

    snapshot = {
        'seed': conf.get('common_seed', 42),
        'stage': 'first_batch',
        'batch_size': bs,
        'train_dataset_size': train_dataset_size,
        'expected_randperm_first5': _perm_first5,
        'max_seq_len': int(max(payload[next(iter(payload))].shape[1], 0)) if payload else 0,
        'n_unique_targets': int(len(set(target.tolist()))),
        'slice_lengths_first20': [int(x) for x in seq_lens.tolist()[:20]],
        'slice_lengths_hash': hash(tuple(int(x) for x in seq_lens.tolist())),
        'raw_client_ids_first20': raw_client_ids[:20] if raw_client_ids_ok else None,
        'raw_client_ids_unique_sorted_first20': (
            sorted(set(raw_client_ids))[:20] if raw_client_ids_ok and raw_client_ids else None
        ),
        'slice_lengths_stats': {
            'min': int(seq_lens.min().item()),
            'max': int(seq_lens.max().item()),
            'mean': float(seq_lens.float().mean().item()),
            'median': float(np.median(seq_lens.numpy())),
            'p25': float(np.percentile(seq_lens.numpy(), 25)),
            'p75': float(np.percentile(seq_lens.numpy(), 75)),
            'p95': float(np.percentile(seq_lens.numpy(), 95)),
        },
        'enumerated_targets_first20': [int(x) for x in target.tolist()[:20]],
        'feature_stats_padded': {},
        'feature_stats_valid': {},
        'spot_check': _batch_spot_check(payload, seq_lens, target, feature_keys),
    }
    for key in feature_keys:
        if key not in payload:
            continue
        padded_stats, valid_stats = _batch_feature_stats(payload[key], seq_lens)
        snapshot['feature_stats_padded'][key] = padded_stats
        snapshot['feature_stats_valid'][key] = valid_stats

    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    logger.info(f'First-batch snapshot saved to "{snapshot_path}"')

    # --- restore RNG state so fit_model sees the SAME first batch as captured ---
    random.setstate(py_state)
    np.random.set_state(np_state)
    _torch.set_rng_state(torch_state)
    if cuda_states is not None:
        _torch.cuda.set_rng_state_all(cuda_states)
    if loader_gen_state is not None:
        train_loader.generator.set_state(loader_gen_state)


def save_data_snapshot(train_data, valid_data, conf):
    stats_path = conf['stats.path']
    snapshot_path = os.path.join(os.path.dirname(stats_path), 'data_snapshot.json')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    col_id = conf['dataset'].get('col_id', 'client_id')
    feature_keys = (
        list(conf['params.trx_encoder.embeddings'].keys())
        + list(conf['params.trx_encoder.numeric_values'].keys())
    )
    all_feature_keys = feature_keys + ['event_time']

    def get_ids(data):
        raw = [rec.get(col_id, rec.get('customer_id', rec.get('installation_id'))) for rec in data]
        return sorted(_to_int_id(x) for x in raw)

    train_ids = get_ids(train_data)
    valid_ids = get_ids(valid_data)

    train_order_ids = [_to_int_id(rec.get(col_id, rec.get('customer_id', rec.get('installation_id')))) for rec in train_data]

    _probe = 31992
    _window = train_order_ids[max(0, _probe - 2): _probe + 3]

    snapshot = {
        'seed': conf.get('common_seed', 42),
        'n_train': len(train_data),
        'n_valid': len(valid_data),
        'train_order_first10': train_order_ids[:10],
        'train_order_at_31992': train_order_ids[_probe] if len(train_order_ids) > _probe else None,
        'train_order_around_31992': _window,
        'train_ids_first20': train_ids[:20],
        'valid_ids_first20': valid_ids[:20],
        'train_ids_sorted_hash': hash(tuple(train_ids)),
        'valid_ids_sorted_hash': hash(tuple(valid_ids)),
        'train_seq_len': _seq_len_stats(train_data),
        'valid_seq_len': _seq_len_stats(valid_data),
        'train_feature_stats': _feature_stats(train_data, all_feature_keys),
        'valid_feature_stats': _feature_stats(valid_data, all_feature_keys),
        'train_spot_check': _spot_check_sequences(train_data, col_id, feature_keys),
        'valid_spot_check': _spot_check_sequences(valid_data, col_id, feature_keys),
    }

    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    logger.info(f'Data snapshot saved to "{snapshot_path}"')


def _flatten_seq(module):
    """Recursively flatten an nn.Sequential to a list of leaf modules."""
    out = []
    if isinstance(module, torch.nn.Sequential):
        for child in module.children():
            out.extend(_flatten_seq(child))
    else:
        out.append(module)
    return out


def _layer_output_stats_coles(x):
    """Stats for either a coles PaddedBatch (payload: tensor (B,T,D)) or Tensor (B,H).

    Mirrors EBES `_layer_output_stats` so JSONs can be diffed key-by-key. Note
    that EBES uses (T,B,D) layout while coles uses (B,T,D); we report shape as
    [T, B, D] in both for consistency.
    """
    if hasattr(x, 'payload') and not isinstance(x.payload, dict):
        payload = x.payload.detach().float().cpu().numpy()
        seq_lens_t = x.seq_lens
        seq_lens = (
            seq_lens_t.detach().cpu().numpy()
            if hasattr(seq_lens_t, 'detach')
            else np.asarray(seq_lens_t)
        )
        B, T, D = payload.shape
        padded = {
            "shape": [T, B, D],
            "min": float(payload.min()),
            "max": float(payload.max()),
            "mean": float(payload.mean()),
            "std": float(payload.std()),
        }
        valid_chunks = []
        for b in range(B):
            sl = int(seq_lens[b])
            if sl > 0:
                valid_chunks.append(payload[b, :sl, :])
        if valid_chunks:
            valid = np.concatenate(valid_chunks, axis=0)
            valid_stats = {
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
            }
        else:
            valid_stats = padded
        d_show = min(8, D)
        sl0 = int(seq_lens[0]) if B > 0 else 0
        spot = {
            "slice0_t0_first8": (
                [float(v) for v in payload[0, 0, :d_show].tolist()] if B > 0 else []
            ),
            "slice0_tlast_first8": (
                [float(v) for v in payload[0, max(0, sl0 - 1), :d_show].tolist()]
                if B > 0 else []
            ),
        }
        return {"kind": "Seq", "padded": padded, "valid": valid_stats, "spot": spot}

    arr = x.detach().float().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    B, H = arr.shape
    d_show = min(8, H)
    out = {
        "kind": "Tensor",
        "shape": [B, H],
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "spot": {
            "slice0_first8": (
                [float(v) for v in arr[0, :d_show].tolist()] if B > 0 else []
            ),
            "slice4_first8": (
                [float(v) for v in arr[min(4, B - 1), :d_show].tolist()] if B > 4 else []
            ),
        },
    }
    row_norms = np.linalg.norm(arr, axis=1)
    out["row_norms_first5"] = [float(v) for v in row_norms[:5].tolist()]
    out["row_norms_stats"] = {
        "min": float(row_norms.min()),
        "max": float(row_norms.max()),
        "mean": float(row_norms.mean()),
    }
    return out


def save_forward_snapshot(model, train_loader, conf):
    """Snapshot-3: param init stats + forward output stats per layer on first batch.

    Mirrors EBES `save_forward_snapshot` so the resulting JSONs are directly
    comparable. Captures (in `data_snapshot_forward.json`):
      - per-parameter init stats (mean/std/min/max/norm/shape/dtype/numel) so we
        can verify EBES and coles initialise to the same weights at iter 0.
      - forward output of each leaf layer of the (possibly nested) Sequential
        on the FIRST training batch. For an `rnn_model` the leaves are
        TrxEncoder -> RnnEncoder -> LastStepEncoder -> L2Normalization, which
        line up 1:1 with EBES's Batch2Seq -> GRU -> TakeLastHidden -> L2Normalization.

    Wraps everything in `model.eval()` + `torch.no_grad()` so global RNG is not
    consumed. The DataLoader iter consumes loader-side state, so we save and
    restore Python/numpy/torch/CUDA/loader.generator RNG states (same pattern
    as save_first_batch_snapshot) to keep fit_model reproducible.
    """
    import torch as _torch

    stats_path = conf['stats.path']
    snapshot_path = os.path.join(os.path.dirname(stats_path), 'data_snapshot_forward.json')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = _torch.get_rng_state()
    cuda_states = _torch.cuda.get_rng_state_all() if _torch.cuda.is_available() else None
    loader_gen_state = (
        train_loader.generator.get_state() if train_loader.generator is not None else None
    )

    seed = conf.get('common_seed', 42)

    snapshot = {"seed": seed, "stage": "first_forward"}

    param_stats = {}
    param_names = []
    for name, p in model.named_parameters():
        param_names.append(name)
        pf = p.detach().float()
        param_stats[name] = {
            "shape": list(p.shape),
            "numel": int(p.numel()),
            "mean": float(pf.mean().item()),
            "std": float(pf.std().item()) if p.numel() > 1 else 0.0,
            "min": float(pf.min().item()),
            "max": float(pf.max().item()),
            "norm": float(pf.norm().item()),
            "dtype": str(p.dtype),
        }
    snapshot["param_names"] = param_names
    snapshot["param_names_hash"] = hash(tuple(param_names))
    snapshot["n_total_params"] = int(sum(p.numel() for p in model.parameters()))
    snapshot["param_stats"] = param_stats

    padded_batch, target = next(iter(train_loader))

    device = next(model.parameters()).device
    if hasattr(padded_batch, 'to'):
        padded_batch = padded_batch.to(device)

    was_training = model.training
    model.eval()
    intermediates = []
    with _torch.no_grad():
        x = padded_batch
        for layer in _flatten_seq(model):
            x = layer(x)
            intermediates.append((type(layer).__name__, x))
    if was_training:
        model.train()

    seq_lens = padded_batch.seq_lens
    if hasattr(seq_lens, 'detach'):
        seq_lens_np = seq_lens.detach().cpu().numpy()
    else:
        seq_lens_np = np.asarray(seq_lens)
    snapshot["batch_meta"] = {
        "B": int(seq_lens_np.shape[0]),
        "lengths_first5": [int(v) for v in seq_lens_np[:5].tolist()],
        "device": str(device),
    }

    forward = {}
    for i, (lname, out) in enumerate(intermediates):
        forward[f"layer_{i}_{lname}"] = _layer_output_stats_coles(out)
    snapshot["forward"] = forward

    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    logger.info(f'Forward snapshot saved to "{snapshot_path}"')

    random.setstate(py_state)
    np.random.set_state(np_state)
    _torch.set_rng_state(torch_state)
    if cuda_states is not None:
        _torch.cuda.set_rng_state_all(cuda_states)
    if loader_gen_state is not None:
        train_loader.generator.set_state(loader_gen_state)


def prepare_data(conf):
    data = read_data_gen(conf['dataset.train_path'])
    data = tqdm(data)
    if 'max_rows' in conf['dataset']:
        data = islice(data, conf['dataset.max_rows'])
    data = prepare_embeddings(data, conf, is_train=True)
    data = shuffle_client_list_reproducible(conf, data)
    data = list(data)
    if 'client_list_keep_count' in conf['dataset']:
        data = data[:conf['dataset.client_list_keep_count']]

    seed = conf.get('common_seed', 42)
    rng = np.random.default_rng(seed)
    valid_ix = rng.choice(len(data), size=int(len(data) * conf['dataset.valid_size']), replace=False)
    valid_ix = set(valid_ix.tolist())

    logger.info(f'Loaded {len(data)} rows. Split in progress...')
    train_data = [rec for i, rec in enumerate(data) if i not in valid_ix]
    valid_data = [rec for i, rec in enumerate(data) if i in valid_ix]

    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    col_id = conf['dataset'].get('col_id', 'client_id')
    if len(train_data) > 31992:
        rec = train_data[31992]
        logger.info(f'COLES_PROBE [end of prepare_data] train_data id(list)={id(train_data)} '
                    f'train_data[31992][{col_id}]={rec.get(col_id)!r} '
                    f'event_time_len={len(rec.get("event_time", []))}')

    return train_data, valid_data


def create_data_loaders(conf):
    train_data, valid_data = prepare_data(conf)

    col_id = conf['dataset'].get('col_id', 'client_id')
    if len(train_data) > 31992:
        logger.info(f'COLES_PROBE [before save_data_snapshot] id(train_data)={id(train_data)} '
                    f'train_data[31992][{col_id}]={train_data[31992].get(col_id)!r}')

    save_data_snapshot(train_data, valid_data, conf)

    if len(train_data) > 31992:
        logger.info(f'COLES_PROBE [after  save_data_snapshot] id(train_data)={id(train_data)} '
                    f'train_data[31992][{col_id}]={train_data[31992].get(col_id)!r}')

    seed = conf.get('common_seed', 42)

    train_dataset = SplittingDataset(
        train_data,
        split_strategy.create(**conf['params.train.split_strategy']),
        seed=seed,
        col_id=col_id,
    )
    train_dataset = TargetEnumeratorDataset(train_dataset)
    train_dataset = ConvertingTrxDataset(train_dataset)
    train_dataset = DropoutTrxDataset(train_dataset, trx_dropout=conf['params.train.trx_dropout'],
                                      seq_len=conf['params.train.max_seq_len'],
                                      seed=seed, col_id=col_id)

    if conf['params.train'].get('all_time_shuffle', False):
        train_dataset = AllTimeShuffleMLDataset(train_dataset)
        logger.info('AllTimeShuffle used')

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_splitted_rows,
        num_workers=conf['params.train'].get('num_workers', 0),
        batch_size=conf['params.train.batch_size'],
        generator=torch.Generator().manual_seed(seed),
    )

    if len(train_data) > 31992:
        # Walk the chain like save_first_batch_snapshot does and verify it lands on train_data
        ds_probe = train_loader.dataset
        chain = [type(ds_probe).__name__]
        while hasattr(ds_probe, 'core_dataset'):
            ds_probe = ds_probe.core_dataset
            chain.append(type(ds_probe).__name__)
        while hasattr(ds_probe, 'delegate'):
            ds_probe = ds_probe.delegate
            chain.append(type(ds_probe).__name__)
        while hasattr(ds_probe, 'base_dataset'):
            ds_probe = ds_probe.base_dataset
            chain.append(type(ds_probe).__name__)
        is_same = ds_probe is train_data
        logger.info(f'COLES_PROBE [after  DataLoader build] chain={chain} '
                    f'unwrapped is train_data?={is_same} '
                    f'unwrapped[31992][{col_id}]='
                    f'{ds_probe[31992].get(col_id) if isinstance(ds_probe, list) and len(ds_probe) > 31992 else "N/A"!r} '
                    f'train_data[31992][{col_id}]={train_data[31992].get(col_id)!r}')

    valid_dataset = SplittingDataset(
        valid_data,
        split_strategy.create(**conf['params.valid.split_strategy'])
    )
    valid_dataset = TargetEnumeratorDataset(valid_dataset)
    valid_dataset = ConvertingTrxDataset(valid_dataset)
    valid_dataset = DropoutTrxDataset(valid_dataset, trx_dropout=0.0,
                                      seq_len=conf['params.valid.max_seq_len'])
    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        collate_fn=collate_splitted_rows,
        num_workers=conf['params.valid'].get('num_workers', 0),
        batch_size=conf['params.valid.batch_size'],
    )

    return train_loader, valid_loader


def run_experiment(model, conf):
    import time
    start = time.time()

    stats_file = conf['stats.path']
    params = conf['params']

    train_loader, valid_loader = create_data_loaders(conf)

    # Snapshot-2: capture first training batch (after slicing/dropout/shuffle).
    # RNG state is saved/restored inside, so fit_model below sees the same
    # first batch we captured.
    save_first_batch_snapshot(train_loader, conf)

    # Snapshot-3: model param init stats + per-layer forward output on first
    # batch. Same save/restore-RNG pattern as snapshot-2 so fit_model is not
    # disturbed.
    save_forward_snapshot(model, train_loader, conf)

    sampling_strategy = get_sampling_strategy(params)
    loss = get_loss(params, sampling_strategy)

    valid_metric = {'BatchRecallTop': BatchRecallTop(k=params['valid.split_strategy.split_count'] - 1)}
    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_handlers = []
    if 'checkpoints' in conf['params.train']:
        checkpoint = CheckpointHandler(
            model=model,
            **conf['params.train.checkpoints']
        )
        train_handlers.append(checkpoint)

    metric_values = fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metric,
                              train_handlers=train_handlers)

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        save_dir = os.path.dirname(conf['model_path.model'])
        os.makedirs(save_dir, exist_ok=True)

        m_encoder = model[0] if conf['model_path.only_encoder'] else model

        torch.save(m_encoder, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

    results = {
        'exec-sec': exec_sec,
        'Recall_top_K': metric_values,
    }

    if conf.get('log_results', True):
        update_model_stats(stats_file, params, results)


def main(args=None):
    conf = get_conf(args)

    seed = conf.get('common_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Global seed set to {seed}')

    model_f = ml_model_by_type(conf['params.model_type'])
    model = model_f(conf['params'])

    return run_experiment(model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()
