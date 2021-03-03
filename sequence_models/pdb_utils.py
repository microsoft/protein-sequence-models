import numpy as np
import scipy
from scipy.spatial.distance import squareform, pdist


def get_dihedrals(a, b, c, d):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)


def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v * w, axis=1)

    return np.arccos(x)


def process_coords(coords):
    N = np.array(coords['N'])
    Ca = np.array(coords['CA'])
    C = np.array(coords['C'])

    # recreate Cb given N,Ca,C
    nres = len(N)
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    # Cb-Cb distance matrix
    dist = squareform(pdist(Cb))
    np.fill_diagonal(dist, np.nan)
    indices = [[i for i in range(nres) if i != j] for j in range(nres)]
    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i]]).T
    idx0 = idx[0]
    idx1 = idx[1]
    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega = np.zeros((nres, nres)) + np.nan
    omega[idx0, idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta = np.zeros((nres, nres)) + np.nan
    theta[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi = np.zeros((nres, nres)) + np.nan
    phi[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    return dist, omega, theta, phi
