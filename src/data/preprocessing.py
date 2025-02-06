import random
from collections.abc import Iterable

import numpy as np
import torch as T
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import default_collate

from src.data.cluster import batch_kt_cluster


def collate_and_transform(
    batch: Iterable[dict],
    do_default_collate: bool = True,
    transforms: list[callable] | None = None,
) -> dict:
    """Collate the batch and apply the transforms.

    Why this not lightning's on_before_batch_transfer?
    This still runs inside the pytorch multiprocessing pool for data loading.
    Thus it runs asynchonously for each batch being prepared.
    """
    if do_default_collate:
        batch = default_collate(batch)
    if transforms is not None:
        for transform in transforms:
            batch = transform(batch)
    return batch


def tokenize_batch(
    jet_dict: dict[T.Tensor],
    token_fn: nn.Module,
) -> dict:
    """Add token versions of the constituents to the jet_dict."""
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]
    out = token_fn.predict(csts[mask].T.contiguous()).long()
    jet_dict["tokens"] = T.zeros(mask.shape, dtype=T.long)
    jet_dict["tokens"][mask] = out
    return jet_dict


def preprocess_batch(
    jet_dict: dict[T.Tensor],
    cst_fn: BaseEstimator,
    jet_fn: BaseEstimator,
) -> dict:
    """Preprocess a batch of jets already stored as pytorch tensors."""
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]
    jets = jet_dict["jets"]

    # Pad the csts with zeros to match what the cst_fn expects
    if (feat_diff := cst_fn.n_features_in_ - csts.shape[-1]) > 0:
        zeros = np.zeros((csts.shape[:-1] + (feat_diff,)), dtype=csts.dtype)
        csts = np.concatenate((csts, zeros), axis=-1)
    csts[mask] = T.from_numpy(cst_fn.transform(csts[mask])).float()
    if feat_diff > 0:
        csts = csts[:, :-feat_diff]  # Remove the padding
    jet_dict["csts"] = csts

    jets = T.from_numpy(jet_fn.transform(jets)).float()
    jet_dict["jets"] = jets

    return jet_dict


def mask_batch(
    jet_dict: dict,
    mask_fraction: float = 0.4,
    key: str = "null_mask",
) -> dict:
    """Applies a masking function of a batch of jets.

    Will add a new key to the jet_dict with the locations of the new mask.
    """
    null_mask = T.stack([
        mask_jet(mask, mask_fraction=mask_fraction) for mask in jet_dict["mask"]
    ])
    jet_dict[key] = null_mask
    return jet_dict


def cluster_mask_batch(
    jet_dict: dict,
    mask_fraction: float = 0.5,
    R: float = 0.1,
    p: float = -1.0,
    key: str = "null_mask",
) -> dict:
    """Clusters the jets using the (anti)-kt algorithm and maskes some sub-jets."""
    csts = jet_dict["csts"].numpy()
    mask = jet_dict["mask"]
    null_mask = T.zeros_like(mask, dtype=T.bool)

    # Cluster the jets
    _, idxes, num_subjets = batch_kt_cluster(csts, R, p)

    # Loop through the batch
    for b_idx in range(csts.shape[0]):
        sjets = list(range(num_subjets[b_idx]))
        to_mask = int(mask_fraction * num_subjets[b_idx])
        random.shuffle(sjets)
        for sj in sjets[:to_mask]:
            null_mask[b_idx][idxes[b_idx] == sj] = True
    jet_dict[key] = null_mask
    return jet_dict


def mask_jet(
    mask: T.Tensor,
    mask_fraction: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly drop a fraction of the jet based on the total number of constituents."""
    if seed is not None:
        T.manual_seed(seed)
    max_drop = int(T.floor(mask_fraction * mask.sum()))
    null_mask = T.zeros_like(mask, dtype=T.bool)

    # Exit now if we are not dropping any nodes
    if max_drop == 0:
        return null_mask

    # Generate a random score per node, the lowest frac will be killed
    rand = T.rand(len(mask))
    rand[~mask] = 9999
    drop_idx = T.argsort(rand)[:max_drop]
    null_mask[drop_idx] = True

    return null_mask
