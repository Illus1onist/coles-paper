# coding: utf-8
import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SplittingDataset(Dataset):
    def __init__(self, base_dataset, splitter, seed=None, col_id='client_id'):
        self.base_dataset = base_dataset
        self.splitter = splitter
        self.seed = seed
        self.col_id = col_id
        self.epoch = 0  # updated each epoch by PrepareEpoch handler

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        # Extract actual client_id for per-client deterministic seeding.
        # Falls back to positional idx when col_id is absent.
        client_id = row.get(self.col_id, row.get('customer_id', row.get('installation_id', idx)))

        indexes = self.splitter.split(local_date, client_id=client_id, seed=self.seed, epoch=self.epoch)
        data = [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data


class SeveralSplittingsDataset(Dataset):
    def __init__(self, base_dataset, splitters):
        self.base_dataset = base_dataset
        self.splitters = splitters

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        data = []
        for splitter in self.splitters:
            indexes = splitter.split(local_date)
            data += [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data
