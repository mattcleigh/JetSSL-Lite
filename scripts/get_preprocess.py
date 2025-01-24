"""Create preprocessors for the constituents and jets using JetClass."""

import argparse

import rootutils
import torch as T
from joblib import dump
from sklearn.preprocessing import QuantileTransformer
from torchpq.clustering import KMeans

root = rootutils.setup_root(search_from=".", pythonpath=True)

import logging

import numpy as np

from src.data.mappable import MapDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess constituents and jets.")
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10_000,
        help="Number of clusters for KMeans.",
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=500,
        help="Number of quantiles for QuantileTransformer.",
    )
    parser.add_argument(
        "--num_jets",
        type=int,
        default=1000_000,
        help="Number of jets to load.",
    )
    parser.add_argument(
        "--num_csts",
        type=int,
        default=64,
        help="Number of constituents.",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/srv/fast/share/rodem/JetClassH5/train_100M_combined.h5",
        help="Path to the dataset file.",
    )
    return parser.parse_args()


# Create the datasets
def main():
    args = get_args()

    log.info("Loading the dataset")
    data = MapDataset(
        file_path=args.file_path,
        num_jets=args.num_jets,  # Roughtly 30M constituents
        num_csts=args.num_csts,
    )

    log.info("Loading pure arrays of the constituents")
    csts = data.data_dict["csts"][data.data_dict["mask"]]
    jets = data.data_dict["jets"]
    log.info(f"Loaded {len(csts)} constituents and {len(jets)} jets")

    log.info("Ignoring neutral impact parameters for the neutral constituents")
    jc_id = data.data_dict["csts_id"][data.data_dict["mask"]]
    is_neut = (jc_id == 0) | (jc_id == 2)
    csts[is_neut, 3:] = np.nan

    log.info("Fitting the quantile transformer for the constituents")
    cst_qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=args.n_quantiles,
        subsample=None,
    )
    cst_qt.fit(csts)
    dump(cst_qt, root / f"resources/cst_quantiles_{args.num_csts}.joblib")

    log.info("Fitting the quantile transformer for the jets")
    jet_qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=args.n_quantiles,
        subsample=None,
    )
    jet_qt.fit(jets)
    dump(jet_qt, root / "resources/jet_quantiles.joblib")

    log.info("Fitting the KMeans tokenizer for the constituents")
    csts = np.nan_to_num(csts)  # Change back the NaNs to 0
    csts = cst_qt.transform(csts)  # We fit the KMeans on the transformed data
    csts_tensor = T.from_numpy(csts).T.contiguous().to("cuda")
    kmeans = KMeans(n_clusters=args.n_clusters, max_iter=300, verbose=10)
    kmeans.fit(csts_tensor)
    kmeans.to("cpu")
    T.save(kmeans, root / f"resources/kmeans_{args.num_csts}.pkl")


if __name__ == "__main__":
    main()
