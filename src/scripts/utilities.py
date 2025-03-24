"""
*******************************************************************************
Copyright (C) [2025] [ATTOP project]

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License, version 3,
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License, version 3,
along with this program. If not, see <https://www.gnu.org/licenses/>.
*******************************************************************************

*******************************************************************************
This file contains functions that are used in other modules.
  
  Morgane Vacher, Vincent Delmas (2023)

*******************************************************************************
"""

import pickle
from typing import List, Any, Tuple, Union
import numpy as np
from scripts.constants import *


def get_idx(file_lines: List[str], string: str) -> List:
    """Finds indexes of the lines where a string appears.

    Args:
        file_lines (List[str]): a List of strings.
        string (str): string to find in the file lines.
    
    Returns:
        List of indeces.
    """
    return [i for (i, line) in enumerate(file_lines) if string in line]


def get_column(lines: List[str], column: int) -> str:
    """Get a particular column from a block of lines.
    
    Args:
        lines (List[str]): A list of strings.
        column (int): The index of the column to extract (0-indexed).
    """
    return " ".join([line.split()[column] for line in lines])


def get_col_array(lines: List[str], col: int) -> np.array:
    """Extracts a specific column from a list of strings and returns it as a NumPy array.

    Args:
        lines (List[str]): A list of strings.
        col (int): The index of the column to extract (0-indexed).

    Returns:
        np.array: A NumPy array containing the elements from the specified column.
                  The dtype of the array will be float if all elements could be
                  converted to float, otherwise it will be string (object dtype).
    """
    try:
        data = np.array([line.split()[col] for line in lines]).astype(float)
    except:
        data = np.array([line.split()[col] for line in lines])

    return data


def gs(X: np.array, row_vecs: bool = True, normalize: bool = True) -> np.array:
    """ Orthogonalizes vectors (rows) using the Gram-Schmidt method.
    
    Args:
        X (np.array): Input matrix.
        row_vecs (bool, optional): Flag indicating if rows of X are vectors.
        Defaults to True, meaning rows are vectors. If False, columns are vectors.
        normalize (bool, optional): Flag indicating if the orthogonal vectors should be normalized.
        Defaults to True.

    Returns:
        np.array: Orthogonalized matrix.
        If `row_vecs` is True, rows are orthogonal.
        If `row_vecs` is False, columns are orthogonal.
    """
    if not row_vecs:
        X = X.T

    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))

    # Normalize
    if normalize:
        Y = np.diag(1/np.linalg.norm(Y, axis=1)).dot(Y)

    return Y if row_vecs else Y.T


def get_data_nm(nb_atoms: int, nb_nm: int, filename: str) -> Tuple:
    """Given a Gaussian/Molcas frequency calculation output file,
    returns the reference geometry [Bohr] and matrices
    to convert from cartesian [Bohr] to mass-frequency scaled normal modes as used in vMCG.

    Args:
        nb_atoms (int): The number of atoms in the molecule.
        nb_nm (int): The number of normal modes to extract.
        filename (str): Path to the Gaussian/Molcas output file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
            A tuple containing:
              - geom_bohr (np.ndarray): Reference geometry in Bohr units.
              - cart2nm (np.ndarray): Matrix to convert from Cartesian coordinates to
                mass-frequency scaled normal modes.
              - nm2cart (np.ndarray): Matrix to convert from mass-frequency scaled normal modes
                to Cartesian coordinates.
              - nm_gs (np.ndarray): Gram-Schmidt orthogonalized normal modes.
              - symbol (List[str]): List of atomic symbols.

    Raises:
        Exception: If the file is not a Gaussian or Molcas output file.
    """
    # Open file and save lines
    with open(filename) as fp:
        file_lines = fp.read().splitlines()

    # Identity type of files: Gaussian or Molcas
    Gaussian, Molcas = False, False
    if len(get_idx(file_lines, "Entering Gaussian System")) != 0:
        Gaussian = True
    elif len(get_idx(file_lines, "This copy of MOLCAS")) != 0 or len(get_idx(file_lines, "This run of MOLCAS")) != 0:
        Molcas = True

    # Get atomic symbols to deduce masses [AMU]
    if Gaussian:
        try:
            idx_geom = get_idx(file_lines, "Standard orientation:")[0] + 5
        except IndexError:
            idx_geom = get_idx(file_lines, "Input orientation:")[0] + 5
        symbol = get_col_array(file_lines[idx_geom: idx_geom + nb_atoms], 1)
    elif Molcas:
        idx_geom = get_idx(file_lines, "Cartesian Coordinates")[0] + 4
        symbol = list(" ".join([line[15] for line in file_lines[idx_geom: idx_geom + nb_atoms]]))[::2]
    else:
        raise Exception("Only works with Gaussian or Molcas files.")

    # Get nb cartesian coordinates
    nb_cart = nb_atoms * 3

    # Add atomic Mass
    masses = np.zeros(nb_cart)
    for c in range(nb_cart):
        masses[c] = ELEMENTS_ATOMIC_MASS.get(symbol[c//3],
                                             [k for k, v in ELEMENTS_ATOMIC_MASS.items() if int(v/2) == c//3])

    # Get origin geometry - geoms in [Angstroms]
    geom = np.zeros(nb_cart)
    for i in range(nb_atoms):
        if Gaussian:
            geom[i * 3: 3 + i * 3] = [float(c) for c in file_lines[idx_geom + i].split()[3: 6]]
        elif Molcas:
            geom[i * 3: 3 + i * 3] = [float(c) for c in file_lines[idx_geom + i].split()[5: 8]]

    # Geom [bohr]
    geom_bohr = geom * ANG_TO_BOHR

    # Read frequencies [cm-1] and normal modes from output file
    frequency = np.zeros(nb_nm)
    nm = np.zeros((nb_nm, nb_cart))
    if Gaussian:
        idx_nm = get_idx(file_lines, "Frequencies")[0]

        # Normal modes are printed 5 by 5
        for i in range(nb_nm // 5):
            frequency[0 + i * 5: 5 + i * 5] = [float(f) for f in file_lines[idx_nm].split()[2:]]
            idx_nm += 5
            nm[0 + i * 5: 5 + i * 5] = [get_col_array(file_lines[idx_nm: idx_nm + nb_cart], 3+n) for n in range(5)]
            idx_nm += 2 + nb_cart

        # If not multiple of 5 -> save rest
        if nb_nm % 5 != 0:
            i = nb_nm // 5
            frequency[0 + i * 5:] = [float(f) for f in file_lines[idx_nm].split()[2:]]
            idx_nm += 5
            nm[0 + i * 5:] = [get_col_array(file_lines[idx_nm: idx_nm + nb_cart], 3+n) for n in range(nb_nm % 5)]
            idx_nm += 2 + nb_cart
    elif Molcas:
        idx_nm = get_idx(file_lines, "Frequency")[0]

        # Normal modes are printed 6 by 6
        for i in range(nb_nm // 6):
            line_freq = file_lines[idx_nm].split()[1:]
            frequency[6*i: 6+i*6] = [(-float(f[1:]) if f[0] == "i" else float(f)) for f in line_freq]
            idx_nm += 5

            nm[6*i: 6+i*6] = [get_col_array(file_lines[idx_nm: idx_nm + nb_cart], 2+n) for n in range(6)]
            idx_nm += 4 + nb_cart

        # If not multiple of 6: save rest
        if nb_nm % 6 != 0:
            i = nb_nm // 6
            frequency[6*1:] = [float(f) for f in file_lines[idx_nm].split()[1:]]
            idx_nm += 5

            nm[i*6:] = [get_col_array(file_lines[idx_nm: idx_nm + nb_cart], 2+n) for n in range(nb_nm % 6)]
            idx_nm += 5 + nb_cart

    # Convert frequencies to Hartree
    freq_hartree = abs(frequency * INV_CM_TO_HARTREE)

    # Re-mass scale normal modes because Gaussian and Molcas output files are rubbish!
    nm_true = nm * np.sqrt(masses)

    # Normalize normal modes
    nm_true = (nm_true.T / np.linalg.norm(nm_true, axis=1)).T

    # Gram-Schmidt orthogonalize normal modes
    nm_gs = gs(nm_true)

    # Dimensionless normal modes -> scale by sqrt(mass * freq)
    # Matrix to go from cart to normal modes
    cart2nm = ((nm_gs * (np.sqrt(masses * AMU_TO_ME))).T * np.sqrt(freq_hartree)).T

    # Matrix to go from normal modes to cart
    nm2cart = (nm_gs / (np.sqrt(masses * AMU_TO_ME))).T / np.sqrt(freq_hartree)

    return geom_bohr, cart2nm, nm2cart, nm_gs, symbol


def trans_back(geom: np.array, geom_ref: np.array, cart2s: np.array) -> np.array:
    """Converts cartesian coordinates to mass-frequency scaled normal modes as used in vMCG.

    Args:
        geom (np.array): Cartesian coordinates.
        geom_ref (np.array): Reference Cartesian coordinates.
        cart2s (np.array): Transformation matrix from Cartesian to scaled normal modes.

    Returns:
        np.array: Mass-frequency scaled normal modes (3N).
    """
    return np.dot(cart2s, (geom - geom_ref))


def trans_for(geom: np.array, geom_ref: np.array, s2cart: np.array) -> np.array:
    """Converts from mass-frequency scaled normal modes to cartesian coordinates as used in vMCG.

    Args:
        geom (np.array): Mass-frequency scaled normal modes.
        geom_ref (np.array): Reference Cartesian coordinates.
        s2cart (np.array): Transformation matrix from scaled normal modes to Cartesian.

    Returns:
        np.array: Cartesian coordinates.
    """
    return geom_ref + np.dot(s2cart, geom)


def angle(p1: np.array, p0: np.array, p2: np.array) -> Union[float, np.array]:
    """Calculates the angle between three points given in Cartesian coordinates.

    Args:
        p1 (np.array): Coordinates of the first point (1x3 or Nx3 array).
        p0 (np.array): Coordinates of the vertex point (1x3 or Nx3 array).
        p2 (np.array): Coordinates of the second point (1x3 or Nx3 array).

    Returns:
        float: Angle in degrees.
    """
    v1 = p1 - p0
    v2 = p2 - p0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    return np.degrees(np.arccos(np.dot(v1, v2)/norm1/norm2))


def dihedral(p0: np.array, p1: np.array, p2: np.array, p3: np.array) -> Union[float, np.array]:
    """Calculates the dihedral angle between four points given in cartesian coordinates.
    Praxeolitic formula 1 sqrt, 1 cross product.

    Args:
        p0 (np.array): Coordinates of the first point (1x3 or Nx3 array).
        p1 (np.array): Coordinates of the second point (1x3 or Nx3 array).
        p2 (np.array): Coordinates of the third point (1x3 or Nx3 array).
        p3 (np.array): Coordinates of the fourth point (1x3 or Nx3 array).

    Returns:
        float: Dihedral angle in degrees.
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 so it does not influence magnitude of vector rejections that come next
    b1 /= np.linalg.norm(b1)

    # Vector rejections
    # v = projection of b0 onto plane perpendicular to b1 = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1 = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # Angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))


def pickle_save(fname: str, data: Any, **kwargs: Any):
    """Wrapper to save to a pickled file.

    Args:
        fname (str): Path to save the `.pickle` file.
        data (Any): Data to be pickled and saved.
        **kwargs (Any): Keyword arguments to be passed to ``pickle.dump``.
    
    Returns:
        None. It writes the `.pickle` file.
    """
    print("Creating pickle file...")
    filehandler = open(fname, "wb")
    pickle.dump(data, filehandler, **kwargs)
    filehandler.close()
    print(f"Pickle saved: {fname}\n")


def pickle_load(fname: str) -> Any:
    """Wrapper to load pickled file.

    Args:
        fname (str): Path to the `.pickle` file to load.

    Returns:
        Any: The loaded data from the `.pickle` file.

    Raises:
        Exception: If there is an error during file loading or `.pickle` reading.
    """
    try:
        print("Loading pickle...")
        with open(fname, "rb") as fp:
            data = pickle.load(fp)
        print("Data loaded!\n")
    except Exception as e:
        raise Exception(f"{fname}: {e}")

    return data


def kabsch_algorithm(to_rotate: np.array, ref: np.array, weights: np.array = None) -> np.array:
    """Applies the Kabsch algorithm to align a point cloud to a reference.

    The Kabsch algorithm finds the optimal rotation and translation to minimize the RMSD
    (Root Mean Square Deviation) between two sets of points. This function takes flattened
    arrays representing 3D point clouds and returns the aligned point cloud as a flattened array.

    Args:
        to_rotate (np.ndarray): Point cloud to be aligned. Should be a 1D NumPy array
            representing a flattened list of 3D coordinates. Shape should be (N,) where N is
            a multiple of 3.
        ref (np.ndarray): Reference point cloud. Should be a 1D NumPy array representing
            a flattened list of 3D coordinates. Shape should be (N,) where N is a multiple of 3.
        weights (np.ndarray, optional): Weights for each point in the point clouds. If provided,
            the algorithm will use the weighted center of mass instead of the geometric center.
            Should be a 1D NumPy array of the same length as the number of points (N/3).
            Defaults to None, which means geometric center is used.

    Returns:
        np.ndarray: Aligned point cloud as a flattened NumPy array. The shape will be the same
            as the input `to_rotate` array (N,).
    """
    to_rotate = to_rotate.reshape(len(to_rotate)//3, 3)
    ref = ref.reshape(len(ref)//3, 3)

    # Calculate mean or weigthed mean (COM)
    if weights is not None:
        to_rotate_mean = np.sum(to_rotate * weights[:, np.newaxis], axis=0) / np.sum(weights)
        ref_mean = np.sum(ref * weights[:, np.newaxis], axis=0) / np.sum(weights)
    else:
        to_rotate_mean = to_rotate.mean(axis=0)
        ref_mean = ref.mean(axis=0)

    # Centroid
    to_rotate_c = to_rotate - to_rotate_mean
    ref_c = ref - ref_mean

    # Covariance matrix
    H = to_rotate_c.T.dot(ref_c)

    # Singular Value Decompostion
    U, S, Vt = np.linalg.svd(H)

    # Transpose
    V = Vt.T

    # Rotation matrix
    R = V.dot(U.T)

    # Translation vector
    t = ref_mean - R.dot(to_rotate_mean)

    # Flatted aligned
    aligned = (R.dot(to_rotate.T)).T + t

    return aligned.ravel()


def save_as_xyz(name: str, data: np.array, atoms: Union[np.array, List]):
    """Saves atomic coordinates to an `.xyz` file.

    Args:
        name (str): The root of the name for the `.xyz` file.
        data (np.array): A NumPy array of atomic coordinates. Each row should represent
            the coordinates (x, y, z) of an atom.
        atoms (Union[np.array, List]): A list or NumPy array of atomic symbols.
            The length of this list should be equal to the number of rows in `data`.

    Returns:
        None. It writes the `.xyz` file.
    """
    with open(f"data_{name}.xyz", "w") as fp:
        fp.write(f"{len(atoms)}\n")
        fp.write("\n")
        for i, row in enumerate(data):
            fp.write(f"{atoms[i]}\t" + "\t".join([str(v) for v in row]) + "\n")
