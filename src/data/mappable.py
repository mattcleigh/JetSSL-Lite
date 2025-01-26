"""For loading in jets from HDF files and creating a mappable dataset."""

import logging
from functools import partial

import h5py
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import collate_and_transform
from src.data.utils import (
    CST_FEATURES,
    JET_FEATURES,
)

log = logging.getLogger(__name__)


class MapDataset(Dataset):
    """Loads a collection of jets from HDF files and creates a mappable dataset."""

    def __init__(
        self,
        file_path: str,
        jet_features: list | None = None,
        cst_features: list | None = None,
        num_jets: int | None = None,
        num_csts: int | None = None,
    ) -> None:
        super().__init__()
        if jet_features is None:
            jet_features = JET_FEATURES
        if cst_features is None:
            cst_features = CST_FEATURES
        self.jet_features = jet_features
        self.cst_features = cst_features

        self.data_dict = {}
        with h5py.File(file_path, mode="r") as f:
            for key in jet_features:
                if key in f:
                    self.data_dict[key] = f[key][:num_jets]
            for key in cst_features:
                if key in f:
                    self.data_dict[key] = f[key][:num_jets, :num_csts]

        self.num_jets = self._get_num_jets(num_jets)
        self.num_csts = self._get_num_csts(num_csts)
        log.info(
            f"Loaded {self.num_jets} jets "
            f"with {self.num_csts} constituents"
            f"from {file_path}"
        )

    def __len__(self) -> int:
        return self.num_jets

    def __getitem__(self, idx: int) -> tuple:
        return {k: v[idx] for k, v in self.data_dict.items()}

    def _get_num_jets(self, num_jets: int | None) -> int:
        file_len = self.data_dict[self.jet_features[0]].shape[0]
        if num_jets is None:
            return file_len
        return min(file_len, num_jets)

    def _get_num_csts(self, num_csts: int | None) -> int:
        file_len = self.data_dict[self.cst_features[0]].shape[1]
        if num_csts is None:
            return file_len
        return min(file_len, num_csts)


class CWolaDataset(Dataset):
    """A mappable dataset that loads signal and background jets and mixes the labels."""

    def __init__(
        self,
        num_sig: int = 1_000,
        num_bkg: int = 100_000,
        sig_file: str = "TTBar",
        bkg_file: str = "ZJetsToNuNu",
        **kwargs,
    ) -> None:
        self.sig = MapDataset(file_path=sig_file, num_jets=num_sig, **kwargs)
        self.bkg = MapDataset(file_path=bkg_file, num_jets=num_bkg, **kwargs)
        self.num_sig = len(self.sig)
        self.num_bkg = len(self.bkg)
        log.info(f"Loadded {self.num_sig} signal and {self.num_bkg} background jets.")

    def __len__(self) -> int:
        return self.num_sig + self.num_bkg

    def __getitem__(self, idx: int) -> tuple:
        if idx < len(self.sig):
            sample = self.signal[idx]
            sample["cwola_labels"] = 1
            sample["labels"] = 1
        else:
            sample = self.background[idx - self.n_signal]
            sample["cwola_labels"] = idx % 2
            sample["labels"] = 0
        return sample


class MapModule(LightningDataModule):
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
        self.data_config = data_config

        # Initialise the validation set now to get a sample
        self.valid_set = MapDataset(self.val_path, **data_config)

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        if stage in {"fit", "train"}:
            self.train_set = MapDataset(self.train_path, **self.data_config)
        if stage in {"predict", "test"}:
            self.test_set = MapDataset(self.test_path, **self.data_config)

    def get_dataloader(self, dataset: Dataset, flag: str) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=flag == "train",
            drop_last=flag == "train",
            collate_fn=partial(collate_and_transform, transforms=self.transforms),
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set, "train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.valid_set, "valid")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_set, "test")

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def on_before_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """Apply any transforms to the batch."""
        if self.transforms is not None:
            for transform in self.transforms:
                batch = transform(batch)
        return batch

    def get_data_sample(self) -> tuple:
        """Get a data sample to initialise the network with the right dimensions."""
        return next(iter(self.valid_set))

    def get_n_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return self.n_classes
