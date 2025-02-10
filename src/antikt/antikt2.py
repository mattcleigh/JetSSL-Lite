import math

import numba
import numpy as np


@numba.njit
def d_ij(part_i: np.ndarray, part_j: np.ndarray, R: float) -> float:
    """Calculate the kt metric between two particles."""
    if part_i[0] == 0 or part_j[0] == 0:
        return 1e9
    ki = part_i[0] ** (-2)
    kj = part_j[0] ** (-2)
    deta = part_i[1] - part_j[1]
    dphi = part_i[2] - part_j[2]
    delta2 = deta**2 + dphi**2
    return min(ki, kj) * delta2 / (R**2)


@numba.njit
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


@numba.njit
def build_dij(csts: np.ndarray, R: float) -> np.ndarray:
    N = csts.shape[0]
    dij = np.zeros((N, N), dtype=np.float64) + 1e9
    for i in range(N):
        if csts[i][0] != 0:
            for j in range(i):
                if csts[j][0] != 0:
                    dij[i][j] = d_ij(csts[i], csts[j], R)
    return dij


@numba.njit
def build_dij_arr(csts: np.ndarray, R: float) -> tuple:
    """Get the nearest neighbour and the distance to it for each particle."""
    N = csts.shape[0]
    dij = np.zeros(N, dtype=np.float64) + 1e9
    n_idx = np.zeros(N, dtype=np.int64) - 1
    for i in range(N):
        _update_neighbour(i, dij, n_idx, csts, R)

    return dij, n_idx


@numba.njit
def _update_neighbour(
    i: int,
    dij: np.ndarray,
    n_idx: np.ndarray,
    csts: np.ndarray,
    R: float,
) -> None:
    """Update the nearest neighbour stored in the n_idx array for particle i.

    Can only consider something as its neighbour if the neighbour comes after itself.
    """
    dij[i] = 1e9
    n_idx[i] = -1
    if csts[i][0] == 0:
        return
    for j in range(i + 1, dij.shape[0]):
        if csts[j][0] != 0:  # Only consider particles with non-zero Pt
            dist = d_ij(csts[i], csts[j], R)
            if dist < dij[i]:
                dij[i] = dist
                n_idx[i] = j


@numba.njit
def _consider_updating_neighbour(
    i: int,
    dij: np.ndarray,
    n_idx: np.ndarray,
    csts: np.ndarray,
    R: float,
) -> None:
    """Consider updating all particle's neighbour information after i has changed."""
    for k in range(i):  # Can only consider i its neighbour if it comes before i
        # Check if the distance to new i is smaller than the current distance
        dist_i = d_ij(csts[i], csts[k], R)
        if dist_i < dij[k]:
            dij[k] = dist_i
            n_idx[k] = i
        # If it previously considered i as its neighbour then it also is updated
        elif n_idx[k] == i:
            _update_neighbour(k, dij, n_idx, csts, R)


@numba.njit
def _remove_particle(
    i: int,
    dib: np.ndarray,
    dij: np.ndarray,
    n_idx: np.ndarray,
    csts: np.ndarray,
    R: float,
) -> None:
    """Remove a particle from all arrays after it has been converted/absorbed."""
    csts[i] = 0
    dib[i] = 1e9
    dij[i] = 1e9
    _consider_updating_neighbour(i, dij, n_idx, csts, R)


@numba.njit
def _update_particle(
    i: int,
    dib: np.ndarray,
    dij: np.ndarray,
    n_idx: np.ndarray,
    csts: np.ndarray,
    R: float,
) -> None:
    """Update all information pertaining to particle i after its properties changed."""
    dib[i] = csts[i, 0] ** (-2)
    # _update_neighbour(i, dij, n_idx, csts, R)  # Is called already in _remove??
    _consider_updating_neighbour(i, dij, n_idx, csts, R)


@numba.njit
def build_dib(csts: np.ndarray) -> np.ndarray:
    return csts[:, 0] ** (-2)


@numba.njit
def _cluster(
    csts: np.ndarray,
    jets: np.ndarray,
    dij: np.ndarray,
    idxes: np.ndarray,
    n_idx: np.ndarray,
    dib: np.ndarray,
    R: float,
) -> int:
    jet_idx = 0
    N = csts.shape[0]
    c_idxes = np.arange(N, dtype=np.int64)

    while N > 0:
        # Find the minimum dmin in both arrays
        min_dij_idx = np.argmin(dij)
        min_dib_idx = np.argmin(dib)
        min_dij = dij[min_dij_idx]
        min_dib = dib[min_dib_idx]

        # If the smallest value is the dij - particles must be merged
        if min_dij < min_dib:
            i = min_dij_idx
            j = n_idx[i]  # The neighbour of i
            _combine(csts[i], csts[j])  # Combine into position i
            _remove_particle(j, dib, dij, n_idx, csts, R)
            _update_particle(i, dib, dij, n_idx, csts, R)
            c_idxes[c_idxes == j] = i  # Update the indices
            N -= 1

        # Otherwise the particle must be converted to a jet
        else:
            i = min_dib_idx
            jets[jet_idx] = csts[i]  # Move the particle to the jet array
            idxes[c_idxes == i] = jet_idx  # All in cluster get same index
            _remove_particle(i, dib, dij, n_idx, csts, R)
            jet_idx += 1
            N -= 1
    return jet_idx


@numba.njit
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
    dij, n_idx = build_dij_arr(csts, R)
    dib = build_dib(csts)

    # Run the cluster method until all csts are converted to jets
    num_jets = _cluster(csts, jets, dij, idxes[mask], n_idx, dib, R)

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
        dij, n_idx = build_dij_arr(csts, R)
        dib = build_dib(csts)

        # Run the cluster method until all csts are converted to jets
        num_jets[b] = _cluster(csts, jets[b], dij, idxes[mask], n_idx, dib, R)

    return jets, idxes, num_jets
