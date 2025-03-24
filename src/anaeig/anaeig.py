"""Script for the analysis of the PCA eigenvectors.
Visualisation of the PCA eigenvectors in Cartesian coordinates.
Initial version Alessandro 02/02/25.
TODO Get rid of the hardcoded constant for the displacement along eigenvectors.
     Docstrings. """

import os
import argparse
import pickle
from typing import Any, NoReturn, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" --- PARAMETERS -------------------------------------------------------------------------------------- """

# Conversion from [Bohr] to [Angstrom]
BOHR_TO_ANG = 0.529177249
U_TO_AMU = 1. / 5.4857990943e-4  # Conversion from g/mol to amu
MASSES = {'H': 1.007825 * U_TO_AMU,
          'He': 4.002603 * U_TO_AMU,
          'Li': 7.016004 * U_TO_AMU,
          'Be': 9.012182 * U_TO_AMU,
          'B': 11.009305 * U_TO_AMU,
          'C': 12.000000 * U_TO_AMU,
          'N': 14.003074 * U_TO_AMU,
          'O': 15.994915 * U_TO_AMU,
          'F': 18.998403 * U_TO_AMU,
          'Cr': 51.940512 * U_TO_AMU}

""" ----------------------------------------------------------------------------------------------------- """


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


def plot_explained_var(eigval: np.array, explvar: np.array, savefig: bool=False) -> NoReturn:
    """ Plots the explained variance as a function of the number of eigenvectors.

    Args:
        Eigenvalues (eigval): np.array
        Explained variance (explvar): np.array
        savefig: bool

    Returns:
        plot and if savefig a .png
    """
    fig, ax = plt.subplots()

    ax.bar(range(len(eigval)), eigval / np.sum(eigval), alpha=0.5,
           align='center', color='#004e92', label='Individual explained variance')
    
    csum_explvar = np.cumsum(explvar)

    ax.step(range(len(csum_explvar)), csum_explvar, where='mid',
            color='#8b2a44', label='Cumulative explained variance')

    ax.set_xlabel('Principal component index', fontsize=14)

    ax.legend(fontsize=14, loc='best')
    ax.set_ylabel('Explained variance ratio', fontsize=14)

    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    if savefig:
        plt.savefig('explained_var.png', dpi=300)
    else:
        pass
    
    plt.show()


def read_xyz_file(filename: str) -> Tuple[np.array, List]:
    """
    Reads an .xyz file and extracts atomic symbols and coordinates.

    Args:
        filename (str): The path to the .xyz file.

    Returns:
        Tuple[np.array, List]: 
            - A NumPy array of shape (N,3) containing atomic coordinates.
            - A list of atomic symbols.
    """
    coordinates = []
    symbols = []

    with open(filename) as file:
        next(file), next(file)
        for line in file:
            if line.strip():
                symbol, x, y, z = line.split()
                coordinates.append([float(x), float(y), float(z)])
                symbols.append(symbol)

    return np.asarray(coordinates), symbols


def write_trj_xyz(filename_xyz: str, at_names: List, geom: np.array):
    """
    Writes an .xyz trajectory file from a geometries stored as a (M,3N) NumPy array
    where M is the number of frames.

    Args:
        filename_xyz (str): The name of the .xyz file to append data to.
        at_names (List[str]): A list of atomic symbols corresponding to the atoms.
        geom (np.array): A NumPy array of shape (N,3) representing atomic coordinates.

    Returns:
        None: The function writes directly to the file and does not return anything.
    """
    with open(filename_xyz, 'a') as xyz:
        xyz.write(f'{len(at_names)}\n\n')
        for i in range(geom.shape[0]):
            xyz.write(f'{at_names[i]} \t {geom[i][0]} \t {geom[i][1]} \t {geom[i][2]} \n')
    
    return None


def proj_along_pcs(pca_pickle, first_pc:int, last_pc: int, traj_data: str):
    # Load CSV file into a DataFrame
    df = pd.read_csv(f"{traj_data}")
    
    # Convert DataFrame to NumPy array
    featurized_data = df.to_numpy()
    
    # Projection
    w = pca_pickle.transform(featurized_data)
    np.savetxt("projections.txt", w[:, first_pc-1:last_pc], header=f"PC{first_pc}-{last_pc} projections")

    return None


def filt_along_pc_new(pca_pickle, filename_xyz: str, at_names: List, traj_data: str, eigvec_nb: int):
    """Filters trajectory data along a principal component and writes the filtered geometries to an .xyz file.

    This function projects trajectory data onto a principal component, reconstructs
    the filtered coordinates, and saves them in an .xyz trajectory file.

    Args:
        pca_pickle: A PCA object with `transform`, `mean_`, `nm2cart`, and `geom_ref_bohr` attributes.
        filename_xyz (str): The name of the output .xyz file.
        at_names (List): A list of atomic symbols corresponding to the atoms.
        traj_data (str): The path to the CSV file containing trajectory data.
        eigvec_nb (int): The index of the principal component to filter along.

    Returns:
        None: The function writes directly to a file and does not return anything.
    """
    # Load CSV file into a DataFrame
    df = pd.read_csv(f"{traj_data}")
    
    # Convert DataFrame to NumPy array
    featurized_data = df.to_numpy()
    
    # Projection
    w = pca_pickle.transform(featurized_data)
    
    # Select component
    filt = np.outer(w[:, eigvec_nb], eigvec[eigvec_nb, :]) + pca_pickle.mean_
    filt_cart = filt @ pca_pickle.nm2cart.T + pca_pickle.geom_ref_bohr
    
    # Write the filtered Cartesian coordinates to a file
    for frame in range(filt_cart.shape[0]):
        write_trj_xyz(f"{filename_xyz}", at_names, filt_cart[frame].reshape(-1, 3) * BOHR_TO_ANG)

    return None


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Analyze a PCA pickle file.\n\
                                     Plot the explained variance as a function of the eigenvectors number.\n\
                                     Filter the trajectory along the selected PC.")
    parser.add_argument("-f", "--file", required=True, help="Path to the PCA .pickle file")
    
    parser.add_argument("--plot", action="store_true", help="Plot explained variance")
    
    parser.add_argument("--filt", action="store_true", help="Filter along the selected PC")
    parser.add_argument("--eignum", help="Selected PC for --filt option")
    parser.add_argument("-o", "--outname", help="Name of the .xyz output file from --filt")
    
    parser.add_argument("--proj", action="store_true", help="Projection of an .xyz trajectory onto the selected PCs")
    parser.add_argument("--xyz", help=".xyz trajectory file to project")
    parser.add_argument("--first", help="First eigenvector to consider for the projection")
    parser.add_argument("--last", help="Last eigenvector to consider for the projection")


    args = parser.parse_args()
    filename = args.file
    plot = args.plot
    filt = args.filt
    eignum = args.eignum
    outname = args.outname
    proj = args.proj
    xyz2proj = args.xyz
    first_pc = int(args.first)
    last_pc = int(args.last)

    # Check dependencies: --savefig requires --plot
    if args.eignum and not args.filt:
        parser.error("--eignum requires --filt. Use: --filt --eignum")
    # Check dependencies: --proj requires --xyz, --first, --last
    if args.proj and not args.xyz and not args.first and not args.last:
        parser.error("--proj requires --xyz, --first, and --last. Add the necessary information to perform projection.")

    pca_pickle = pickle_load(filename)

    # Eigenvectors
    eigvec = pca_pickle.components_
    # Eigenvalues
    eigval = pca_pickle.explained_variance_
    # Explained variance (eigenvalues / sum(eigenvalues))
    explvar = pca_pickle.explained_variance_ratio_

    savefig = False

    if plot:
        plot_explained_var(eigval, explvar, savefig)
        if savefig:
            print("Plot saved to explained_var.png")
        else:
            print("Plot not saved. Set savefig to True if you want to save a .png.")
    else:
        print("Plot not shown. Enable --plot if you want to plot.")

    if filt:
        ref_geom, symbols = read_xyz_file("init_geom_0001.xyz")
        traj_data = "data_saved.csv"
        filt_along_pc_new(pca_pickle, outname, symbols, traj_data, int(eignum))

    # Check dependencies: --proj requires --xyz, --first, --last
    if proj:
        traj_data = "data_saved.csv"
        proj_along_pcs(pca_pickle, first_pc, last_pc, traj_data)
    