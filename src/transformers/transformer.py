"""
*******************************************************************************
Copyright (C) [2025] [ATTOP project]

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License, version 2.1.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
for more details at <http://www.gnu.org/licenses/>.
*******************************************************************************

*******************************************************************************
File 

  Vincent Delmas, Alessandro Nicola Nardi (2025)

*******************************************************************************
"""

import pickle
import sys
from typing import Tuple, Any
import numpy as np


""" --- PARAMETERS -------------------------------------------------------------------------------------- """

# Conversion from [Bohr] to [Angstrom]
BOHR_TO_ANG = 0.529177249

""" ----------------------------------------------------------------------------------------------------- """


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
        Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)

    return Y if row_vecs else Y.T


def pickle_load(fname: str) -> Any:
    """Wrapper to load a pickled file.

    Args:
        fname (str): The path to the pickled file to be loaded.

    Returns:
        Any: The data loaded from the pickle file.
        The type depends on what was saved in the file.

    Raises:
        Exception: If the file cannot be opened or the pickle operation fails,
        an exception with details is raised.
    """
    try:
        print("Loading pickle...")
        with open(fname, "rb") as fp:
            data = pickle.load(fp)
        print("Data loaded!\n")
    except Exception as e:
        raise Exception(f"{fname}: {e}")

    return data


def get_data_and_masses_from_file(file: str) -> Tuple[np.array, np.array]:
    """
    Get the coordinates/forces/velocities and masses from a file given by Molcas.

    Args:
        file (str): Path to the Molcas output file containing data sections.

    Returns:
        Tuple[np.array, np.array]: A tuple containing:
            - data (np.array): A 2D NumPy array of shape (N,3) where N is the number of atoms, 
              representing the coordinates, forces, or velocities.
            - masses (np.array): A 2D NumPy array of shape (N,1) where N is the number of atoms, 
              representing the atomic masses.

    Raises:
        ValueError: If the file is missing required sections `[DATA]` or `[MASS]`.
        Exception: For other unexpected issues while reading or processing the file.
    """
    with open(file) as fp:
        lines = fp.readlines()

    data_start = lines.index("[DATA]\n") + 1
    mass_start = lines.index("[MASS]\n") + 1
    atoms = lines[data_start: mass_start - 1]

    data = np.array([line.split() for line in atoms], dtype=float).reshape(len(atoms), 3)
    masses = np.array(lines[mass_start:], dtype=float).reshape(-1, 1)

    return data, masses


def get_data_masses_and_geom_from_file(file: str) -> Tuple[np.array, np.array, np.array]:
    """
    Get the coordinates/forces/velocities and masses from a file given by Molcas.

    Args:
        file (str): Path to the Molcas output file containing data sections.

    Returns:
        Tuple[np.array, np.array]: A tuple containing:
            - data (np.array): A 2D NumPy array of shape (N,3) where N is the number of atoms,
              representing the coordinates, forces, or velocities.
            - masses (np.array): A 2D NumPy array of shape (N,1) where N is the number of atoms,
              representing the atomic masses.
            - geom (np.array): A 2D NumPy array of shape (N,3) where N is the number of atoms,
              representing the coordinates.

    Raises:
        ValueError: If the file is missing required sections `[DATA]`, `[MASS]` or `[GEOM]`.
        Exception: For other unexpected issues while reading or processing the file.
    """
    with open(file) as fp:
        lines = fp.readlines()

    data_start = lines.index("[DATA]\n") + 1
    mass_start = lines.index("[MASS]\n") + 1
    geom_start = lines.index("[GEOM]\n") + 1
    atoms = lines[data_start: mass_start - 1]

    data = np.array([line.split() for line in atoms], dtype=float).reshape(len(atoms), 3)
    masses = np.array(lines[mass_start: geom_start - 1], dtype=float).reshape(-1, 1)

    coord = lines[geom_start:]
    geom = np.array([line.split() for line in coord], dtype=float).reshape(len(atoms), 3)

    return data, masses, geom


def write_to_file(reduced_data: np.array, output: str):
    """Save the new (reduced) data to a file in a specific format for Molcas.

    This function writes the reduced data to the specified file, where each row
    of the numpy array is written as a tab-separated line. A header `[DATA]` is
    added at the beginning of the file.

    Args:
        reduced_data (np.array): The numpy array containing the data to be saved.
        output (str): The file path where the data should be saved.

    Returns:
        None. It saves a text file.
    """
    with open(output, "w") as fp:
        fp.write("[DATA]\n")
        for row in reduced_data:
            fp.write("\t\t".join([str(v) for v in row]) + "\n")


def apply_variance(pickle: str, data: np.array, pattern: str, masses: np.array) -> np.array:
    """The core function for reducing the data applying the NM variace.

    Args:
        pickle (str): pickle binary file containing the information about the NM variance. 
        data (np.array): the data read by `get_data_and_masses_from_file` function
            under `[DATA]` section from the file by Molcas. This is the object that will be reduced.
        pattern (str): indicates the nature of the object to be reduced: 
            "geom":geometry, "vel":velocity, "force":force.
        masses (np.array): array of the atomic masses.
    
    Returns:
        np.array of the new (reduced) data that will be saved to a file and given to Molcas.
    
    Raise:
        Exception:
            if `pattern` is not recognized.
    """
    # Load nm pickle
    nm = pickle_load(pickle)

    # Calculate masses vector square root
    sqrt_masses = np.sqrt(masses)

    if pattern == "geom":
        print("*** entered 'variance' - 'geom' ***")

        # Transform to Normal Modes ([nm])
        data_nm = np.dot(nm.cart2nm, (data.ravel() - nm.geom_ref_bohr))

        # Set selected [nm] to zero
        for mode in nm.to_remove:
            data_nm[mode] = 0
        print("data nm", data_nm)

        # Transform back to [Bohr] with only the selected nm
        new_data = nm.geom_ref_bohr + np.dot(nm.nm2cart, data_nm)

        # Reshape data
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)
    elif pattern == "vel":
        print("*** entered 'variance' - 'vel' ***")

        # Transform to [nm]
        data_nm = np.dot(nm.cart2nm, data.ravel())

        # Set selected [nm] to zero
        for mode in nm.to_remove:
            data_nm[mode] = 0
        print("data nm", data_nm)

        # Transform back to with only the selected nm
        new_data = np.dot(nm.nm2cart, data_nm)

        # Reshape data
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)
    elif pattern == "force":
        print("*** entered 'variance' - 'force' ***")

        # Mass weighting
        data = data / sqrt_masses

        # Transform to [nm]
        data_nm = np.dot(nm.nm_matrix, data.ravel())

        # Set selected [nm] to zero
        for mode in nm.to_remove:
            data_nm[mode] = 0
        print("data nm", data_nm)

        # Transform back to with only the selected nm
        new_data = np.dot(nm.nm_matrix.T, data_nm)

        # Remove mass-weighting
        new_data = new_data.reshape(len(data.ravel()) // 3, 3) * sqrt_masses

        # Reshape data
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)
    else:
        raise Exception("Pattern not recognized.")

    return new_data


def apply_PCA(pickle: str, data: np.array, pattern: str, masses: np.array) -> np.array:
    """The core function for reducing the data applying the PCA.

    Args:
        pickle (str): pickle binary file (sklearn PCA) containing the information about the PCA. 
        data (np.array): the data read by `get_data_and_masses_from_file` function
            under `[DATA]` section from the file by Molcas. This is the object that will be reduced.
        pattern (str): indicates the nature of the object to be reduced: 
            "geom":geometry, "vel":velocity, "force":force.
        masses (np.array): array of the atomic masses.
    
    Returns:
        np.array of the new (reduced) data that will be saved to a file and given to Molcas.
    
    Raise:
        Exception:
            if `pattern` is not recognized.
    """
    # Calculate masses vector square root
    sqrt_masses = np.sqrt(masses)

    # Load PCA pickle
    pca = pickle_load(pickle)
    proj = pca.components_

    if pattern == "vel" and pca.repr == "cart":
        print("*** entered 'vel' and 'cart' ***")

        # Mass-weight velocity
        data_mw = data * sqrt_masses

        # Mass-weight projection + GS
        weigthed_proj = []
        for row in proj:
            new_row = row.reshape(len(row) // 3, 3) * sqrt_masses
            weigthed_proj.append(new_row.ravel())
        weigthed_proj = gs(np.asarray(weigthed_proj))

        # Perform projection
        reduced = np.dot(weigthed_proj, data_mw.ravel())
        new_data = np.dot(reduced, weigthed_proj).ravel()

        # Reshape
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)

        # Un-Mass-weight velocity vector
        new_data = new_data / sqrt_masses
    elif pattern == "geom" and pca.repr == "cart":
        print("*** entered 'geom' and 'cart' ***")

        # From [Bohr] to [Ang]
        data = data.ravel() * BOHR_TO_ANG

        # Mass-Weight [Ang]
        data_mw = data.reshape(len(data.ravel()) // 3, 3) * sqrt_masses

        # Mass-weight <pca.mean_> already [Ang]
        pca.mean_ = (pca.mean_.reshape(len(data.ravel()) // 3, 3) * sqrt_masses).ravel()

        # Mass-weight <pca.components_> + GS
        weigthed_proj = []
        for row in pca.components_:
            new_row = row.reshape(len(row) // 3, 3) * sqrt_masses
            weigthed_proj.append(new_row.ravel())
        weigthed_proj = gs(np.asarray(weigthed_proj))

        # Change <pca.components_> by mass-weighted ones
        pca.components_ = weigthed_proj

        reduced = pca.transform(data_mw.reshape(1, -1))
        new_data = pca.inverse_transform(reduced).ravel()

        # Un-Mass-Weight [Ang]
        new_data = (new_data.reshape(len(data.ravel()) // 3, 3) / sqrt_masses).ravel()

        # Convert to [Bohr] again
        new_data *= 1 / BOHR_TO_ANG

        # Reshape data
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)
    elif pattern == "force" and pca.repr == "cart":
        print("*** entered 'force' and 'cart' ***")

        # Mass-weight force
        data_mw = data / sqrt_masses

        # Mass-weight <pca.components_> + GS
        weigthed_proj = []
        for row in pca.components_:
            new_row = row.reshape(len(row) // 3, 3) * sqrt_masses
            weigthed_proj.append(new_row.ravel())
        weigthed_proj = gs(np.asarray(weigthed_proj))

        # Change <pca.components_> by mass-weighted ones
        pca.components_ = weigthed_proj

        # Perform projection
        reduced = np.dot(weigthed_proj, data_mw.ravel())
        new_data = np.dot(reduced, weigthed_proj)

        # Reshape
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)

        # Un-Mass-weight force vector
        new_data = new_data * sqrt_masses
    elif pattern == "vel" and pca.repr == "nm":
        print("*** entered 'vel' and 'nm' ***")

        # Mass-weight velocity
        data_mw = data * sqrt_masses

        # From [nm] to [cart] on PCA components + GS
        proj = gs(np.dot(pca.components_, pca.nm2cart.T))

        # Mass-weight projection + GS
        weigthed_proj = []
        for row in proj:
            new_row = row.reshape(len(row) // 3, 3) * sqrt_masses
            weigthed_proj.append(new_row.ravel())
        weigthed_proj = gs(np.asarray(weigthed_proj))

        # Perform projection
        reduced = np.dot(weigthed_proj, data_mw.ravel())
        new_data = np.dot(reduced, weigthed_proj)

        # Reshape
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)

        # Un-Mass-weight velocity vector
        new_data = new_data / sqrt_masses
    elif pattern == "geom" and pca.repr == "nm":
        print("*** entered 'geom' and 'nm' ***")

        # Mass-Weight [Bohr]
        data_mw = (data * sqrt_masses).ravel()

        # From [nm] to [cart] in [Bohr] on <pca.mean_> + mass-weight
        pca.mean_ = np.dot(pca.mean_, pca.nm2cart.T)
        pca.mean_ = (pca.mean_.reshape(len(data.ravel()) // 3, 3) * sqrt_masses).ravel()

        # From [nm] to [cart] on PCA components_ + GS
        proj = gs(np.dot(pca.components_, pca.nm2cart.T))

        # Mass-weight <pca.components_> + GS
        weigthed_proj = []
        for row in proj:
            new_row = row.reshape(len(row) // 3, 3) * sqrt_masses
            weigthed_proj.append(new_row.ravel())
        weigthed_proj = gs(np.asarray(weigthed_proj))

        # Forth and back
        reduced = np.dot((data_mw - pca.mean_).reshape(1, -1), weigthed_proj.T)
        new_data = (np.dot(reduced, weigthed_proj) + pca.mean_).ravel()

        # Reshape data
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)

        # Remove mass-weighting
        new_data = new_data / sqrt_masses
    elif pattern == "force" and pca.repr == "nm":
        print("*** entered 'force' and 'nm' ***")

        # Mass-weight force
        data_mw = data / sqrt_masses

        # From [nm] to [cart] on PCA components + GS
        proj = gs(np.dot(pca.components_, pca.nm2cart.T))

        # Mass-weight projection + GS
        weigthed_proj = []
        for row in proj:
            new_row = row.reshape(len(row) // 3, 3) * sqrt_masses
            weigthed_proj.append(new_row.ravel())
        weigthed_proj = gs(np.asarray(weigthed_proj))

        # Perform projection
        reduced = np.dot(weigthed_proj, data_mw.ravel())
        new_data = np.dot(reduced, weigthed_proj)

        # Reshape
        new_data = new_data.reshape(len(data.ravel()) // 3, 3)

        # Un-Mass-weight force vector
        new_data = new_data * sqrt_masses
    else:
        exit()

    return new_data


if __name__ == "__main__":
    # Ensure correct number of command-line arguments are provided
    if len(sys.argv) != 6:
        print("Error: Please provide all required command-line arguments.")
        sys.exit(1)

    # Fetch arguments from Molcas input
    try:
        file_from_molcas = sys.argv[1]
        filename_to_output = sys.argv[2]
        method = sys.argv[3].lower()
        pattern = sys.argv[4]
        pickle_fn = sys.argv[5]
    except IndexError:
        print("Error: Insufficient number of command-line arguments.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        sys.exit(1)

    # Get data and masses from Molcas .red.out file
    # data, masses = get_data_and_masses_from_file(file_from_molcas)
    data, masses, geom = get_data_masses_and_geom_from_file(file_from_molcas)

    # Apply PCA or NM variance
    if method == "pca":
        reduced_data = apply_PCA(pickle_fn, data, pattern, masses)
    elif method == "var":
        reduced_data = apply_variance(pickle_fn, data, pattern, masses)
    else:
        print("Reducing method name not recognized (options: 'pca', 'var'). Exiting.")
        exit(1)

    # Write out file for Molcas
    write_to_file(reduced_data, filename_to_output)
