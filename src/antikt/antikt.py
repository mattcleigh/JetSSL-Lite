import math

import numba
import numpy as np


@numba.jit(nopython=True)
def d_ij(part_i: np.ndarray, part_j: np.ndarray, R: float) -> float:
    """Calculate the kt metric between two particles."""
    ki = part_i[0] ** (-2)
    kj = part_j[0] ** (-2)
    deta = part_i[1] - part_j[1]
    dphi = part_i[2] - part_j[2]
    delta2 = deta**2 + dphi**2
    return min(ki, kj) * delta2 / (R**2)


@numba.jit(nopython=True)
def _combine(part_i: np.ndarray, part_j: np.ndarray) -> None:
    """Combine two particles into one."""
    px = part_i[0] * math.cos(part_i[2]) + part_j[0] * math.cos(part_j[2])
    py = part_i[0] * math.sin(part_i[2]) + part_j[0] * math.sin(part_j[2])
    pz = part_i[0] * math.sinh(part_i[1]) + part_j[0] * math.sinh(part_j[1])
    pt = math.hypot(px, py)
    p = math.sqrt(px**2 + py**2 + pz**2)
    eta = math.atanh(pz / p)
    phi = math.atan2(py, px)
    part_i[0] = pt
    part_i[1] = eta
    part_i[2] = phi


@numba.jit(nopython=True)
def build_dij(csts: np.ndarray, R: float) -> np.ndarray:
    N = csts.shape[0]
    dij = np.zeros((N, N), dtype=np.float64) + 1e9
    for i in range(N):
        if csts[i][0] != 0:
            for j in range(i):
                if csts[j][0] != 0:
                    dij[i][j] = d_ij(csts[i], csts[j], R)
    return dij


@numba.jit(nopython=True)
def _update_dij(dij: np.ndarray, csts: np.ndarray, i: int, j: int, R: float) -> None:
    """Update the dij array after particle i absorbs particle j."""
    N = dij.shape[0]
    for k in range(N - 1):
        if k < i:
            if csts[k][0] != 0:
                dij[i, k] = d_ij(csts[i], csts[k], R)
        elif csts[k + 1][0] != 0:
            dij[k + 1, i] = d_ij(csts[k + 1], csts[i], R)
    dij[j, :] = 1e9
    dij[:, j] = 1e9


@numba.jit(nopython=True)
def build_dib(csts: np.ndarray) -> np.ndarray:
    return csts[:, 0] ** (-2)


@numba.jit(nopython=True)
def _update_dib(dib: np.ndarray, csts: np.ndarray, i: int, j: int) -> None:
    """Update the dib array after particle i absorbs particle j."""
    dib[i] = csts[i, 0] ** (-2)
    dib[j] = 1e9


@numba.jit(nopython=True)
def min_2d(arr: np.ndarray) -> tuple:
    """Find the min of a 2D array, numpy unravel_index is not supported by numba."""
    min_val = 1e9
    min_idx = (0, 0)
    for i in range(arr.shape[0]):
        for j in range(i):
            if arr[i, j] < min_val:
                min_val = arr[i, j]
                min_idx = (i, j)
    return min_val, min_idx


@numba.jit(nopython=True)
def _cluster(
    csts: np.ndarray,
    jets: np.ndarray,
    idxes: np.ndarray,
    dij: np.ndarray,
    dib: np.ndarray,
    R: float,
) -> int:
    jet_idx = 0
    N = csts.shape[0]
    c_idxes = np.arange(N, dtype=np.int64)

    while N > 0:
        # Find the minimum dmin in both arrays
        min_dij, min_dij_idx = min_2d(dij)
        min_dib_idx = np.argmin(dib)
        min_dij = dij[min_dij_idx]
        min_dib = dib[min_dib_idx]

        # If the smallest value is the dij - particles must be combined
        if min_dij < min_dib:
            i, j = min_dij_idx
            _combine(csts[i], csts[j])  # Combine into position i
            csts[j] = 0  # Remove the j particle from the pool
            c_idxes[c_idxes == j] = i  # Update the indices
            _update_dij(dij, csts, i, j, R)  # Update the dij array
            _update_dib(dib, csts, i, j)  # Update the dib array
            N -= 1

        # Otherwise the particle must be converted to a jet
        else:
            i = min_dib_idx
            jets[jet_idx] = csts[i]  # Move the particle to the jet array
            csts[i] = 0  # Remove it from the pool
            idxes[c_idxes == i] = jet_idx  # All in cluster get same index
            dij[i, :] = 1e9  # Remove the particle from the dij array
            dij[:, i] = 1e9
            dib[i] = 1e9  # Remove the particle from the dib array
            jet_idx += 1
            N -= 1

    return jet_idx


@numba.jit(nopython=True)
def antikt(x: np.ndarray, R: float) -> tuple:
    """Perform anti-kt clustering on a collection of particles.

    x must be a zero padded 2D array containing [Pt, eta, phi] for each particle.
    """
    N, _D = x.shape
    idxes = np.zeros(N, dtype=np.int64) - 1  # Indexes of the jets per particle

    # Select only the particles with non-zero Pt
    mask = x[:, 0] != 0
    csts = x[mask, :3].copy().astype(np.float64)  # Copy to avoid modifying the input
    N = csts.shape[0]  # Update the number of particles

    # Array to hold the output
    jets = np.zeros((N, 3), dtype=np.float64)

    # Initialize the arrays for dib and dij
    dij = build_dij(csts, R)
    dib = build_dib(csts)

    # Run the cluster method until all csts are converted to jets
    num_jets = _cluster(csts, jets, idxes[mask], dij, dib, R)

    # Remove the zero padded jets
    jets = jets[:num_jets]

    return jets, idxes, num_jets


@numba.jit(nopython=True, parallel=True)
def batch_antikt(x: np.ndarray, R: float) -> tuple:
    """Perform anti-kt clustering on a batch of particles."""
    B, N, _D = x.shape
    jets = np.zeros((B, N, 3), dtype=np.float64)
    idxes = np.zeros((B, N), dtype=np.int64) - 1
    num_jets = np.zeros(B, dtype=np.int64)

    # Loop through the batch
    for b in numba.prange(B):
        csts = x[b]
        mask = csts[:, 0] != 0
        csts = csts[mask, :3].copy().astype(np.float64)

        # Initialize the arrays for dib and dij
        dij = build_dij(csts, R)
        dib = build_dib(csts)

        # Run the cluster method until all csts are converted to jets
        num_jets[b] = _cluster(csts, jets[b], idxes[b], dij, dib, R)

    return jets, idxes, num_jets
