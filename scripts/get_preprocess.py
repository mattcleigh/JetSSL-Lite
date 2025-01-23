"""Create preprocessors for the constituents and jets using JetClass."""

import rootutils
from joblib import dump
from sklearn.preprocessing import QuantileTransformer

root = rootutils.setup_root(search_from=".", pythonpath=True)

import logging

import numpy as np

from src.data.mappable import MapDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Create the datasets
def main():
    log.info("Loading the dataset")
    data = MapDataset(
        file_path="/srv/fast/share/rodem/JetClassH5/train_100M_combined.h5",
        num_jets=2_000_000,  # Roughtly 80M constituents
        num_csts=32,
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
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=500,
        subsample=len(csts) + 1,
    )
    qt.fit(csts)
    dump(qt, root / "resources/cst_quantiles.joblib")

    log.info("Fitting the quantile transformer for the jets")
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=500,
        subsample=len(jets) + 1,
    )
    qt.fit(jets)
    dump(qt, root / "resources/jet_quantiles.joblib")


if __name__ == "__main__":
    main()
