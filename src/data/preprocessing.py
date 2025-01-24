import numpy as np
import torch as T
from sklearn.base import BaseEstimator
from torch import nn


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


def mask_jet(
    mask: T.Tensor,
    mask_fraction: float = 0.4,
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
