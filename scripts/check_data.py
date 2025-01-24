"""Check that the data has been prepared correctly by making a few plots."""

import joblib
import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import numpy as np

from mltools.mltools.plotting import plot_multi_hists
from src.data.mappable import MapDataset

# Define the labels and bins for the plots
cst_labels = [
    r"$p_\text{T} [GeV]$",
    r"$\Delta\eta$",
    r"$\Delta\phi$",
    r"$d0$",
    r"$\sigma(d0)$",
    r"$z0$",
    r"$\sigma(z0)$",
]
cst_bins = [
    np.linspace(0, 800, 25),
    np.linspace(-1, 1, 25),
    np.linspace(-1, 1, 25),
    np.linspace(-1000, 1000, 25),
    np.linspace(0, 0.8, 25),
    np.linspace(-1000, 1000, 25),
    np.linspace(0, 2, 25),
]
jet_labels = [
    r"$p_\text{T} [GeV]$",
    r"$\eta$",
    r"$\phi$",
    "Mass [GeV]",
    r"$N_\text{cst}$",
]
jet_bins = [
    np.linspace(0, 1000, 25),
    np.linspace(-3, 3, 25),
    np.linspace(-4, 4, 25),
    np.linspace(0, 400, 25),
    np.linspace(0, 130, 25),
]


# Create the datasets
def main():
    # Load the two datasets
    bt_data = MapDataset(
        file_path="/srv/fast/share/rodem/btag/train.h5",
        num_jets=10_000,
        num_csts=64,  # Will automatically be reduced to the number in file
    )
    jc_data = MapDataset(
        file_path="/srv/fast/share/rodem/JetClassH5/val_5M_combined.h5",
        num_jets=10_000,
        num_csts=64,
    )

    # Load pure arrays of the constituents
    jc_csts = jc_data.data_dict["csts"][jc_data.data_dict["mask"]]
    bt_csts = bt_data.data_dict["csts"][bt_data.data_dict["mask"]]

    # Neutral impact parameters are zero padded, change to nan to ignore in plots
    jc_id = jc_data.data_dict["csts_id"][jc_data.data_dict["mask"]]
    is_neut = (jc_id == 0) | (jc_id == 2)
    jc_csts[is_neut, 3:] = np.nan

    # Plot the constituent features for the two datasets
    for i in range(len(cst_labels)):
        plot_multi_hists(
            data_list=[jc_csts[:, i : i + 1], bt_csts[:, i : i + 1]],
            fig_height=4,
            data_labels=["JetClass", "BTag"],
            bins=cst_bins[i],
            logy=True,
            ignore_nans=True,
            col_labels=[cst_labels[i]],
            legend_kwargs={"loc": "upper right"},
            hist_kwargs=[{"fill": True, "alpha": 0.5}, {"fill": True, "alpha": 0.5}],
            path=root / f"plots/csts_{i}.png",
            do_norm=True,
        )

    # Load pure arrays of the jets
    jc_jets = jc_data.data_dict["jets"]
    bt_jets = bt_data.data_dict["jets"]

    # Plot the constituent features for the two datasets
    for i in range(len(jet_labels)):
        plot_multi_hists(
            data_list=[jc_jets[:, i : i + 1], bt_jets[:, i : i + 1]],
            fig_height=4,
            data_labels=["JetClass", "BTag"],
            bins=jet_bins[i],
            logy=True,
            ignore_nans=True,
            col_labels=[jet_labels[i]],
            legend_kwargs={"loc": "upper right"},
            hist_kwargs=[{"fill": True, "alpha": 0.5}, {"fill": True, "alpha": 0.5}],
            path=root / f"plots/jets_{i}.png",
            do_norm=True,
        )

    # Plot the transformed constituent features for the two datasets
    cst_fn = joblib.load(root / "resources/cst_quantiles_64.joblib")
    jc_csts = cst_fn.transform(jc_csts)
    bt_csts = cst_fn.transform(bt_csts)
    for i in range(len(cst_labels)):
        plot_multi_hists(
            data_list=[jc_csts[:, i : i + 1], bt_csts[:, i : i + 1]],
            fig_height=4,
            data_labels=["JetClass", "BTag"],
            ignore_nans=True,
            logy=True,
            col_labels=[cst_labels[i]],
            legend_kwargs={"loc": "upper right"},
            hist_kwargs=[{"fill": True, "alpha": 0.5}, {"fill": True, "alpha": 0.5}],
            path=root / f"plots/transformed_csts_{i}.png",
            do_norm=True,
        )

    # Plot the transformed jet features for the two datasets
    jet_fn = joblib.load(root / "resources/jet_quantiles.joblib")
    jc_jets = jet_fn.transform(jc_jets)
    bt_jets = jet_fn.transform(bt_jets)
    for i in range(len(jet_labels)):
        plot_multi_hists(
            data_list=[jc_jets[:, i : i + 1], bt_jets[:, i : i + 1]],
            fig_height=4,
            data_labels=["JetClass", "BTag"],
            ignore_nans=True,
            logy=True,
            col_labels=[jet_labels[i]],
            legend_kwargs={"loc": "upper right"},
            hist_kwargs=[{"fill": True, "alpha": 0.5}, {"fill": True, "alpha": 0.5}],
            path=root / f"plots/transformed_jets_{i}.png",
            do_norm=True,
        )


if __name__ == "__main__":
    main()
