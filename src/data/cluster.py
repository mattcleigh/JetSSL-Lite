import math

import numba
import numpy as np


@numba.jit(nopython=True)
def d_iB(part_i: np.ndarray, p: float) -> float:
    """Calculate the distance between a particle and the beam."""
    return part_i[0] ** (2 * p)


@numba.jit(nopython=True)
def d_ij(part_i: np.ndarray, part_j: np.ndarray, R: float, p: float) -> float:
    """Calculate the kt metric between two particles."""
    ki = part_i[0] ** (2 * p)
    kj = part_j[0] ** (2 * p)
    deta = part_i[1] - part_j[1]
    dphi = part_i[2] - part_j[2]
    delta2 = deta**2 + dphi**2
    return min(ki, kj) * delta2 / (R**2)


@numba.jit(nopython=True)
def combine(part_i: np.ndarray, part_j: np.ndarray) -> np.ndarray:
    """Combine two particles into one."""
    px = part_i[0] * math.cos(part_i[2]) + part_j[0] * math.cos(part_j[2])
    py = part_i[0] * math.sin(part_i[2]) + part_j[0] * math.sin(part_j[2])
    pz = part_i[0] * math.sinh(part_i[1]) + part_j[0] * math.sinh(part_j[1])
    pt = math.sqrt(px**2 + py**2)
    p = math.sqrt(px**2 + py**2 + pz**2)
    eta = 0.5 * math.log((p + pz) / (p - pz))
    phi = math.atan2(py, px)
    return np.array([pt, eta, phi])


@numba.jit(nopython=True)
def get_match(ref: np.ndarray, csts: np.ndarray, R: float, p: float) -> int:
    """Find the best match for a reference particle."""
    dib = d_iB(ref, p)
    for i in range(csts.shape[0]):
        if csts[i][0] != 0:
            dij = d_ij(ref, csts[i], R, p)
            if dij < dib:
                return i
    return -1


@numba.jit(nopython=True)
def _merge_all(
    subjets: np.ndarray,
    csts: np.ndarray,
    idxes: np.ndarray,
    cluster_idx: int,
    R: float,
    p: float,
) -> int:
    """Merge all particles into subjets."""
    while True:
        idx = get_match(subjets[cluster_idx], csts, R, p)
        if idx == -1:
            return 0
        subjets[cluster_idx] = combine(subjets[cluster_idx], csts[idx])
        idxes[idx] = cluster_idx
        csts[idx] = 0  # Set to zero to ignore in the future


@numba.jit(nopython=True)
def _cluster(
    csts: np.ndarray, subjets: np.ndarray, idxes: np.ndarray, R: float, p: float
) -> int:
    """Cluster particles into subjets."""
    cluster_idx = 0
    while True:
        ref_idx = np.argmax(csts[:, 0])  # Hardest particle is the reference

        # If the hardest particle Pt is zero, we are done
        if csts[ref_idx][0] == 0:
            return cluster_idx

        # Absorb the reference into the subjet
        subjets[cluster_idx] = csts[ref_idx]
        idxes[ref_idx] = cluster_idx
        csts[ref_idx] = 0  # Set to zero to ignore in the future

        # Continuously try to merge in more particles
        _merge_all(subjets, csts, idxes, cluster_idx, R, p)

        # Move to the next cluster
        cluster_idx += 1


@numba.jit(nopython=True)
def kt_cluster(csts: np.ndarray, R: float, p: float) -> tuple:
    """Perform anti-kt clustering on a collection of particles.

    csts must be a zero padded 2D array containing [Pt, eta, phi] for each particle.
    """
    N, _D = csts.shape
    subjets = np.zeros((N, 3), dtype=np.float64)  # Arrays to hold the outputs
    idxes = np.zeros(N, dtype=np.int64) - 1  # -1 means not clustered
    csts = csts.copy()[:, :3].astype(np.float64)  # Copy to avoid modifying the input
    nc = _cluster(csts, subjets, idxes, R, p)
    return subjets[:nc], idxes, nc


@numba.jit(nopython=True)
def batch_kt_cluster(csts: np.ndarray, R: float, p: float) -> tuple:
    """Perform anti-kt clustering per collection of particles in a batch.

    csts must be a zero padded 3D array containing [Pt, eta, phi, ...].
    """
    B, N, _D = csts.shape
    subjets = np.zeros((B, N, 3), dtype=np.float64)  # Arrays to hold the outputs
    idxes = np.zeros((B, N), dtype=np.int64) - 1  # -1 means not clustered
    num_subjets = np.zeros(B, dtype=np.int64)
    csts = csts.copy()[..., :3].astype(np.float64)  # Copy to avoid modifying the input

    # Loop over all the events in the batch
    for b_idx in range(csts.shape[0]):
        nc = _cluster(csts[b_idx], subjets[b_idx], idxes[b_idx], R, p)
        num_subjets[b_idx] = nc
    return subjets, idxes, num_subjets


# Example usage
if __name__ == "__main__":
    # Create some random particles
    gen = np.random.default_rng(1234)
    csts = gen.random((100, 3))
    csts[:, 0] *= 100  # Set Pt to be between 0 and 100

    # Cluster the particles
    subjets, idxes, num_subjets = kt_cluster(csts, R=0.2, p=-1.0)

    # Print the results
    print("Number of subjets:")
    print(num_subjets)
    print("Subjets:")
    print(subjets)
    print("Indices:")
    print(idxes)
