"""For streaming in one huge HDF file."""

import logging

import h5py
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.utils import CST_FEATURES, JET_FEATURES

log = logging.getLogger(__name__)


def batch_idxes(
    batch_size: int,
    num_jets: int,
    drop_last: bool = False,
    start: int = 0,
) -> list:
    """Construct a generator of batch indexes."""
    drop_last &= num_jets % batch_size != 0
    return list(range(start, num_jets - drop_last * batch_size, batch_size))


class StreamDataset(Dataset):
    """Streams in jets from a single large HDF file.

    Due to the speed of reading from disk which is a bottleneck, we read in a single
    slice, i.e. a batch of jets, at a time.
    """

    def __init__(
        self,
        file_path: str,
        jet_features: list | None = None,
        cst_features: list | None = None,
        num_jets: int | None = None,
        num_csts: int | None = None,
        batch_size: int = 1000,
    ) -> None:
        if jet_features is None:
            jet_features = JET_FEATURES
        if cst_features is None:
            cst_features = CST_FEATURES
        self.jet_features = jet_features
        self.cst_features = cst_features
        self.batch_size = batch_size

        # Open the file and calculate the length
        self.file = h5py.File(file_path, mode="r")

        # Trim the features to those present in the file so we dont do it every time
        self.jet_features = [k for k in self.jet_features if k in self.file]
        self.cst_features = [k for k in self.cst_features if k in self.file]

        self.num_jets = self._get_num_jets(num_jets)
        self.num_csts = self._get_num_csts(num_csts)
        log.info(
            f"Streaming {self.num_jets} jets "
            f"with {self.num_csts} constituents"
            f"from {file_path}"
        )

    def _get_num_jets(self, num_jets: int | None) -> int:
        file_len = self.file[self.jet_features[0]].shape[0]
        if num_jets is None:
            return file_len
        return min(file_len, num_jets)

    def _get_num_csts(self, num_csts: int | None) -> int:
        file_len = self.file[self.cst_features[0]].shape[1]
        if num_csts is None:
            return file_len
        return min(file_len, num_csts)

    def __len__(self) -> int:
        return self.num_jets

    def __getitem__(self, idx: int) -> tuple:
        """Get a batch of jets."""
        self.data_dict = {}
        idx_f = idx + self.batch_size
        for key in self.jet_features:
            self.data_dict[key] = self.file[key][idx:idx_f]
        for key in self.cst_features:
            self.data_dict[key] = self.file[key][idx:idx_f, : self.num_csts]
        return self.data_dict


class StreamModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_path: str,
        val_path: str,
        test_path: str,
        n_classes: int,
        num_workers: int = 6,
        batch_size: int = 1000,
        pin_memory: bool = True,
        transforms: list | None = None,
        **data_config,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.n_classes = n_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.transforms = transforms
        self.data_config = {**data_config, "batch_size": batch_size}
        self.batch_idx = 0

        # Initialise the validation set now to get a sample
        self.valid_set = StreamDataset(self.val_path, **self.data_config)

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        if stage in {"fit", "train"}:
            self.train_set = StreamDataset(self.train_path, **self.data_config)
        if stage in {"predict", "test"}:
            self.test_set = StreamDataset(self.test_path, **self.data_config)

    def get_dataloader(self, dataset: Dataset, flag: str) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            sampler=batch_idxes(
                self.batch_size,
                len(dataset),
                drop_last=flag == "train",
                start=self.batch_idx,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=None,  # dataset returns a batches already!
            collate_fn=None,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set, "train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.valid_set, "val")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_set, "test")

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def on_before_batch_transfer(self, batch: dict, dataloader_idx: int) -> None:
        """Update the last batch index during validation."""
        if self.trainer.validating:
            self.batch_idx = self.trainer.global_step // self.trainer.current_epoch
        if self.transforms is not None:
            for transform in self.transforms:
                batch = transform(batch)
        return batch

    def get_data_sample(self) -> tuple:
        """Get a data sample to help initialise the network."""
        return next(iter(self.valid_set))

    def load_state_dict(self, state_dict: dict) -> None:
        self.batch_idx = state_dict["batch_idx"]

    def state_dict(self) -> dict:
        return {"batch_idx": self.batch_idx}

    def get_n_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return self.n_classes
