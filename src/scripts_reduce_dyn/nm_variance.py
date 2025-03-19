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
Script Overview:
----------------
It performs operations related to normal modes, variance computation, and PCA
container creation.

Functions:
----------
1. compute_and_sort_variance:
    - Computes variance for each column in the data array.
    - Sorts the list of (index, variance) tuples in descending order of variance.
    - Optionally prints sorted variances and corresponding column indices.

2. remove_n_lowest_nm:
    - Removes the N lowest normal modes from the provided NormalModes object.
    - Creates a PCA container for saving the data.
    - Saves the PCA container as a pickle file.

3. create_normal_modes:
    - Loads an <ensemble> and a <NormalModes> object.
    - Uses <FitterReducer> class to create normal modes data from the <ensemble>.
    - Returns the <NormalModes> object and the normal modes data array.

Main:
-----
- Loads an <ensemble> and creates normal modes data using
  <create_normal_modes> function.
- Computes and prints the sorted variance with indices using
  <compute_and_sort_variance> function.
- Removes N lowest normal modes iteratively and saves PCA containers as pickle
  binary files.

  Vincent Delmas (2023)

*******************************************************************************
"""

from typing import Tuple, List
import numpy as np
from sklearn.decomposition import PCA
from scripts.utilities import pickle_load, pickle_save
import scripts.class_TSH as TSH


def compute_and_sort_variance(data: np.array, print_variance: bool = True) -> List[Tuple[int, float]]:
    """Computes the variance for each column in the dataset, sorts them in descending order,
    and optionally prints the variance.

    Args:
        data (np.array): A 2D NumPy array where variance is computed for each column.
        print_variance (bool, optional): If True, prints variance values and cumulative percentages. Defaults to True.

    Returns:
        List[Tuple[int, float]]: A list of tuples containing column indices and
        their corresponding variance, sorted in descending order.
    """
    # Compute variance for each column and store it with original index
    variance_with_index = [(i, np.var(col)) for i, col in enumerate(data.T)]

    # Sort list of (index, variance) tuples in descending order of variance
    variance_with_index.sort(key=lambda x: x[1], reverse=True)

    # Print sorted variances and corresponding column indices
    if print_variance:
        total = sum(var for _, var in variance_with_index)
        expl_var = 0

        print("\n--- VARIANCE PER MODE --------------------------------------")
        for idx, variance in variance_with_index:
            expl_var += variance
            print(f"Mode {idx + 1:02d}\t - \tVariance = {variance:.2f} - Cumulated (%) = {(expl_var / total * 100):.2f}")

        print(f"\nTotal variance: {total:.3f}")
        print("------------------------------------------------------------\n")

    return variance_with_index


def remove_n_lowest_nm(nm: TSH.NormalModes, variance_with_index: np.array, to_remove: int):
    """Removes the N lowest normal modes and creates a PCA container.

    Args:
        nm (TSH.NormalModes): A normal modes object containing transformation matrices and geometry data.
        variance_with_index (np.array): A sorted array of tuples (index, variance) representing variance per mode.
        to_remove (int): The number of lowest-variance normal modes to remove.

    Returns:
        None: The function saves a PCA container as a pickle file.
    """

    modes_removed = [mode for mode, _ in variance_with_index][-to_remove:]
    print(f"Modes removed: {[mode + 1 for mode in modes_removed]}")

    # Create a PCA container (IS ONLY A CONTAINER NO PCA INVOLVED)
    pca = PCA()
    pca.cart2nm = nm.cart2nm
    pca.nm2cart = nm.nm2cart
    pca.nm_matrix = nm.nm_matrix
    pca.geom_ref_bohr = nm.geom_ref_bohr
    pca.to_remove = modes_removed

    # Save as pickle file
    pickle_save(f"container_var_{len(variance_with_index) - len(modes_removed)}_dim_nm.pickle", pca)


def create_normal_modes(ens_path: str, nm_path: str, nb_atoms: int, nb_nm: int) -> Tuple[TSH.NormalModes, np.array]:
    """Creates normal modes for a specified ensemble and returns the normal modes object along with featurized data.

    Args:
        ens_path (str): Path to the ensemble pickle file.
        nm_path (str): Path to the normal modes file.
        nb_atoms (int): Number of atoms in the system.
        nb_nm (int): Number of normal modes.

    Returns:
        Tuple[TSH.NormalModes, np.array]: A tuple containing the NormalModes object and the featurized dataset.
    """
    # Load <ensemble> and <nm> object
    ensemble = pickle_load(ens_path)
    nm = TSH.NormalModes(filename=nm_path, nb_atoms=nb_atoms, nb_nm=nb_nm)

    # Create normal modes and create dataframe with pandas
    fr = TSH.FitterReducer(ensemble)
    fr.featurizer(repr="nm", nm=nm, kabsch=True, COM=True, mass_w=False)

    return nm, fr.featurized_data_


if __name__ == "__main__":
    # Create normal modes
    nm, data = create_normal_modes(
        "../postdoc_data/molecules_data/butyrolactone/full_200_steps/ensemble.pickle",
        "../molecules_data/butyrolactone/OG_CASSCF.freq.output",
        12,
        30,
    )

    # Get variance sorted and with index
    variance_sorted_with_index = compute_and_sort_variance(data, print_variance=True)

    # Remove N lowest normal modes and save container as pickle
    for to_remove in [2, 4]:
        remove_n_lowest_nm(nm, variance_sorted_with_index, to_remove)
