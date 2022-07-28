import gzip
import numpy as np
import scipy
from scipy.spatial.distance import squareform, pdist

from sequence_models.constants import IUPAC_CODES


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


def parse_PDB(x, atoms=["N", "CA", "C"], chain=None):
    """
    input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    xyz, seq, min_resn, max_resn = {}, {}, np.inf, -np.inf
    open_func = gzip.open if x.endswith('.gz') else open
    for line in open_func(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12 : 12 + 4].strip()
                resi = line[17 : 17 + 3]
                resn = line[22 : 22 + 5].strip()
                x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    for resn in range(min_resn, max_resn + 1):
        if resn in seq:
            for k in sorted(seq[resn]):
                seq_.append(IUPAC_CODES.get(seq[resn][k].capitalize(), "X"))
        else:
            seq_.append("X")
        if resn in xyz:
            for k in sorted(xyz[resn]):
                for atom in atoms:
                    if atom in xyz[resn][k]:
                        xyz_.append(xyz[resn][k][atom])
                    else:
                        xyz_.append(np.full(3, np.nan))
        else:
            for atom in atoms:
                xyz_.append(np.full(3, np.nan))

    valid_resn = np.array(sorted(xyz.keys()))
    return np.array(xyz_).reshape(-1, len(atoms), 3), "".join(seq_), valid_resn


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
