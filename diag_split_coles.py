"""Diagnostic: coles-paper side of the split comparison.

Reads the coles-paper age train parquet and applies the exact same logic as
`prepare_data` + `shuffle_client_list_reproducible` in metric_learning.py
(sorted by client_id native type → Python Random.shuffle → np.random.default_rng.choice).

Prints the same fields as diag_split_ebes.py in the EBES repo so they can be
compared directly to find where the two pipelines diverge.

Run from the coles-paper repo root:
    python diag_split_coles.py
    # or override path:
    python diag_split_coles.py experiments/scenario_age_pred/data/train_trx.parquet
"""
import sys
import hashlib
import random as py_random

import numpy as np
import pyarrow.parquet as pq

PARQUET_PATH = "experiments/scenario_age_pred/data/train_trx.parquet"
ID_COL = "client_id"
SEED = 42
VAL_SIZE = 0.05


def read_records(path: str) -> list[dict]:
    """Read parquet as list of per-client dicts (like read_pyarrow_file)."""
    table = pq.read_table(source=path)
    col_names = table.column_names
    records = []
    for rb in table.to_batches():
        col_arrays = [rb.column(i).to_numpy(zero_copy_only=False) for i in range(len(col_names))]
        for row in zip(*col_arrays):
            records.append({n: (np.array(a) if isinstance(a, np.ndarray) else a)
                            for n, a in zip(col_names, row)})
    return records


def get_client_id(rec: dict):
    return rec.get(ID_COL, rec.get('customer_id', rec.get('installation_id')))


def paper_split(records: list[dict], seed: int, val_size: float):
    """Mirror of shuffle_client_list_reproducible + prepare_data."""
    # Sort by native type (matches `sorted(data, key=lambda x: x['client_id'])`)
    data = sorted(records, key=lambda x: get_client_id(x))

    # Python Random shuffle
    py_random.Random(seed).shuffle(data)

    # Val indices
    n = len(data)
    rng = np.random.default_rng(seed)
    val_ix = rng.choice(n, size=int(n * val_size), replace=False)
    val_set = set(val_ix.tolist())

    train_data = [rec for i, rec in enumerate(data) if i not in val_set]
    return data, train_data  # shuffled list + train subset


def stable_hash(ids: list) -> str:
    joined = ",".join(str(x) for x in ids)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def main(path: str = PARQUET_PATH):
    print(f"=== coles-paper split diagnostic ===")
    print(f"Path       : {path}")
    print(f"Seed       : {SEED}, val_size={VAL_SIZE}")
    print()

    records = read_records(path)
    print(f"N_total    : {len(records)}")
    id_sample = get_client_id(records[0])
    print(f"ID type    : {type(id_sample).__name__}  (example: {_to_int(id_sample)})")

    # Sorted order (before shuffle) — lex sort (client_ids are str in coles-paper parquet)
    sorted_records = sorted(records, key=lambda x: get_client_id(x))
    sorted_ids = [str(get_client_id(r)) for r in sorted_records]
    print(f"Sorted IDs (first 10, lex): {[_to_int(x) for x in sorted_ids[:10]]}")
    print(f"Sorted IDs hash           : {stable_hash(sorted_ids)}")
    print()

    shuffled, train_data = paper_split(records, SEED, VAL_SIZE)
    train_ids_str = [str(get_client_id(r)) for r in train_data]
    print(f"N_train    : {len(train_data)}")
    print(f"Train IDs (first 10 in train-list order): {[_to_int(x) for x in train_ids_str[:10]]}")
    train_ids_int = sorted(_to_int(x) for x in train_ids_str)
    print(f"Train IDs (sorted int, first 10)        : {train_ids_int[:10]}")
    print(f"Train sorted IDs hash : {stable_hash(train_ids_int)}")

    # Position 31992 — that's torch.randperm(N, seed=42)[0] for N=44650.
    # If this differs from EBES, the split orderings diverge.
    probe = 31992
    if len(train_ids_str) > probe:
        print()
        print(f"Train ID at position {probe}        : {_to_int(train_ids_str[probe])}")
        print(f"Train IDs around position {probe}    : {[_to_int(x) for x in train_ids_str[probe-2:probe+3]]}")

    # Also show first 5 val choice indices and first 5 shuffled positions for comparison
    n = len(records)
    rng_v = np.random.default_rng(SEED)
    val_pos = rng_v.choice(n, size=int(n * VAL_SIZE), replace=False)
    print()
    print(f"rng.choice first 5    : {val_pos[:5].tolist()}")

    # Shuffle a list of positions to compare with EBES side
    shuffled_positions = list(range(n))
    py_random.Random(SEED).shuffle(shuffled_positions)
    print(f"shuffled_positions[:10]: {shuffled_positions[:10]}")

    # Also show what record sits at shuffled_positions[31992] in sorted_records
    # (this is what EBES iloc-indexes via sorted_df.iloc[shuffled_positions[i]] for the train set)
    sorted_records_local = sorted(records, key=lambda x: get_client_id(x))
    shuf2 = list(range(n))
    py_random.Random(SEED).shuffle(shuf2)
    val_set = set(val_pos.tolist())
    train_shuffled_idx = [i for i in range(n) if i not in val_set]
    print(f"train_shuffled_idx[31992] (i.e. shuffled-array position picked for train slot 31992): "
          f"{train_shuffled_idx[probe] if len(train_shuffled_idx) > probe else 'OOB'}")
    if len(train_shuffled_idx) > probe:
        sp = shuf2[train_shuffled_idx[probe]]
        print(f"sorted_records[shuffled_positions[that idx]] client_id: "
              f"{_to_int(get_client_id(sorted_records_local[sp]))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PARQUET_PATH)
