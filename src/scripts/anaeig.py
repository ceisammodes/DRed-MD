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
          'F': 18.998403 * U_TO_AMU}

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
    """Docstring TODO
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


def write_trj_xyz(filename_xyz, at_names, geom):
    '''Writes .xyz trajectories from a geometry stored as a np.array
    '''
    with open(filename_xyz, 'a') as xyz:
        xyz.write(f'{len(at_names)}\n\n')
        for i in range(geom.shape[0]):
            xyz.write(f'{at_names[i]} \t {geom[i][0]} \t {geom[i][1]} \t {geom[i][2]} \n')


def filt_along_pc(pca_pickle, filename_xyz, ref_geom, at_names, coefficents, eigvec_nb) -> NoReturn:

    nb_atoms = ref_geom.shape[0]
    masses = np.asarray([MASSES[a] for a in at_names])
    sqrt_masses = np.sqrt(masses)
    ref_geom = ref_geom.reshape(3 * nb_atoms)

    # From [nm] to [cart] on PCA components_
    proj = np.dot(pca_pickle.components_, pca_pickle.nm2cart.T)

    for c in coefficents:
        new_geom = ref_geom + 4.0 * c * proj[eigvec_nb]
        new_geom = new_geom.reshape(nb_atoms,3)
        write_trj_xyz(filename_xyz, at_names, new_geom)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Analyze a PCA pickle file.\n\
                                     Plot the explained variance as a function of the eigenvectors number.\n\
                                     Filter the trajectory along the selected PC.")
    parser.add_argument("-f", "--file", required=True, help="Path to the PCA .pickle file")
    parser.add_argument("--plot", action="store_true", help="Plot explained variance")
    parser.add_argument("--filt", action="store_true", help="Filter along the selected PC")
    parser.add_argument("--eignum", help="Selected PC for --filt option")
    parser.add_argument("-o", "--outname", help="Name of the .xyz output file")

    args = parser.parse_args()
    filename = args.file
    plot = args.plot
    filt = args.filt
    eignum = args.eignum
    outname = args.outname

    # Check dependencies: --savefig requires --plot
    if args.eignum and not args.filt:
        parser.error("--eignum requires --filt. Use: --filt --eignum")

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
        # os.remove("filt.xyz")
        coefficents = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, \
                        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        ref_geom, symbols = read_xyz_file("trans_AZM_opt_casscf_6e4o.Opt.xyz")
        filt_along_pc(pca_pickle, outname, ref_geom, symbols, coefficents, int(eignum))