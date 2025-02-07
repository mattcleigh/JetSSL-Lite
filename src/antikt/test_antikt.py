import time

import awkward as ak
import fastjet
import h5py
import numpy as np
import rootutils
import vector

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import src.antikt.antikt as ant1
import src.antikt.antikt2 as ant2

vector.register_awkward()
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.2)


def fastjet_antikt(csts: np.ndarray) -> tuple:
    """Perform anti-kt clustering using FastJet."""
    csts_ak = ak.zip(  # FastjJet requires awkward arrays
        {
            "pt": csts[:, 0],
            "eta": csts[:, 1],
            "phi": csts[:, 2],
            "M": np.zeros_like(csts[:, 0]),
        },
        with_name="Momentum4D",
    )

    # Cluster the jets
    cluster = fastjet.ClusterSequence(csts_ak, JETDEF)
    jets = cluster.inclusive_jets()
    c_idxes = cluster.constituent_index()

    return jets, c_idxes, len(jets)


def fix_fastjet(csts: np.ndarray, jets: np.ndarray, c_idxes: np.ndarray) -> tuple:
    """Unify the fastjet output format with the numba output format."""
    # The indices are the index of the jets - convert
    idxes = np.zeros(csts.shape[0], dtype=np.int64) - 1
    for i, c in enumerate(c_idxes):
        idxes[c] = len(jets) - i - 1

    # Convert both back to numpy
    jets = np.array([(j.pt, j.eta, j.phi) for j in jets], dtype=np.float64)
    return jets[::-1], idxes, len(jets)


def time_loop_fn(fn, csts, *args) -> float:
    ts = time.time()
    for c in csts:
        fn(c, *args)
    return time.time() - ts


def time_batch_fn(fn, csts, *args) -> float:
    ts = time.time()
    fn(csts, *args)
    return time.time() - ts


def main():
    B = 1000
    with h5py.File("/srv/fast/share/rodem/JetClassH5/val_5M_combined.h5", "r") as f:
        csts = f["csts"][:B]

    # Jit compile the numba functions
    ant1.antikt(csts[0], R=0.2)
    ant2.antikt(csts[0], R=0.2)
    ant1.batch_antikt(csts, R=0.2)
    ant2.batch_antikt(csts, R=0.2)

    # Loop through the batch and check that the jet outputs are the same
    for b in range(B):
        c = csts[b]
        m = c[:, 0] != 0

        # FastJet
        j_fj, i_fj, _ = fastjet_antikt(c[m])
        j_fj, _, _ = fix_fastjet(c[m], j_fj, i_fj)

        # Numba
        j_n1, _, _ = ant1.antikt(c[m], R=0.2)
        j_n2, _, _ = ant2.antikt(c[m], R=0.2)

        # Batched Numba
        j_bn1, _, n_bn1 = ant1.batch_antikt(c[None], R=0.2)
        j_bn2, _, n_bn2 = ant2.batch_antikt(c[None], R=0.2)
        j_bn1 = j_bn1[0, : n_bn1[0]]
        j_bn2 = j_bn2[0, : n_bn2[0]]

        # Check that the outputs are the same
        for i, comp in enumerate([j_n1, j_n2, j_bn1, j_bn2]):
            if j_fj.shape[0] != comp.shape[0]:
                print(f"{b}-{i}: Length mismatch")
                continue
            diff = np.max(np.abs(j_fj - comp))
            if diff > 1e-4:
                print(f"{b}-{i}: {diff:.5f}")

    # Time the functions
    t = time_loop_fn(fastjet_antikt, csts)
    print(f"FastJet:   {t:.5f}")
    t = time_loop_fn(ant1.antikt, csts, 0.2)
    print(f"Numba1:    {t:.5f}")
    t = time_loop_fn(ant2.antikt, csts, 0.2)
    print(f"Numba2:    {t:.5f}")
    t = time_batch_fn(ant1.batch_antikt, csts, 0.2)
    print(f"Batch1:    {t:.5f}")
    t = time_batch_fn(ant2.batch_antikt, csts, 0.2)
    print(f"Batch2:    {t:.5f}")


if __name__ == "__main__":
    main()
