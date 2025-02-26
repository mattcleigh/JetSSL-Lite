import argparse

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np

from mltools.mltools.utils import signed_angle_diff
from src.data.root_utils import (
    common_particle_class,
    lifetime_signing,
    read_shlomi_file,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert BTag .root files to a more usable .HDF format"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/srv/beegfs/scratch/groups/rodem/datasets/btag/",
        help="The path to the Shlomi files",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default="/srv/fast/share/rodem/btag/",
        help="The path to save the converted files",
    )
    return parser.parse_args()


def main() -> None:
    """Convert BTag .root files to a more usable .HDF format.

    The output features 15 dimensions.
    Independant continuous
    - 0: pt
    - 1: deta
    - 2: dphi
    - 3: d0val
    - 4: d0err
    - 5: dzval
    - 6: dzerr
    One-hot encoded categorical representing following classes
    - 0: isPhoton
    - 1: isHadron_Neg
    - 2: isHadron_Neutral
    - 3: isHadron_Pos
    - 4: isElectron_Neg
    - 5: isElectron_Pos
    - 6: isMuon_Neg
    - 7: isMuon_Pos
    """
    # Get the arguments
    args = get_args()

    # Make sure the destination path exists
    dest_path = Path(args.dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Get all of the root files in the source path
    source_path = Path(args.source_path)
    root_files = list(source_path.iterdir())

    # Loop over the root files
    for file in root_files:
        print(file)

        # Read the root file
        jets, tracks, labels, vtx_pos, track_type = read_shlomi_file(file)

        # The labels are 0, 4, 5, change them to 0, 1, 2
        labels = np.where(labels == 4, 1, labels)
        labels = np.where(labels == 5, 2, labels)
        labels = np.squeeze(labels)

        # We need to split the tracks into the different groups of information
        csts = tracks[..., :7]
        charge = tracks[..., 7]
        pdgid = tracks[..., 8]
        vtx_id = tracks[..., 9].astype("l")

        # Switch to lifetime signing convention for the impact parameters
        d0, z0 = lifetime_signing(
            d0=csts[..., 3],
            z0=csts[..., 5],
            tracks=csts[..., :3],
            jets=jets[..., :3],
        )
        csts[..., 3] = d0
        csts[..., 5] = z0

        # The csts eta and phi must be centered on the jet
        csts[..., 1] -= jets[:, 1:2]
        csts[..., 2] = signed_angle_diff(csts[..., 2], jets[:, 2:3])

        # Convert the particle class information to the common format
        csts_id = common_particle_class(charge, pdgid)

        # Get a mask based on track pt
        mask = csts[..., 0] > 0

        # The shlomi datasets are not internally shuffled, which is an issue when
        # setting up training sessions on a portion of the data.
        idx = np.random.default_rng().permutation(len(jets))

        # Save the data to an HDF file
        dest_file = dest_path / file.name.replace(".root", ".h5")
        print("--saving to: ", dest_file)
        with h5py.File(dest_file, "w") as f:
            f.create_dataset("jets", data=jets[idx])
            f.create_dataset("labels", data=labels[idx])
            f.create_dataset("csts", data=csts[idx])
            f.create_dataset("csts_id", data=csts_id[idx])
            f.create_dataset("vtx_id", data=vtx_id[idx])
            f.create_dataset("vtx_pos", data=vtx_pos[idx])
            f.create_dataset("mask", data=mask[idx])
            f.create_dataset("track_type", data=track_type[idx])


if __name__ == "__main__":
    main()
