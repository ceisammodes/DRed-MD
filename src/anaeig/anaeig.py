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
Script for the analysis and the visualisation of the PCA eigenvectors.
  
  Alessandro Nicola Nardi (2023)

*******************************************************************************
"""

import os
import argparse
import pickle
from typing import Any, NoReturn, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator


""" --- PLOT PARAMETERS --------------------------------------------------------------------------------- """

# Set global font to Arial
matplotlib.rcParams['font.family'] = 'Arial'

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
    """Orthogonalizes vectors (rows) using the Gram-Schmidt method.
    
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
    """Plots the explained variance as a function of the number of eigenvectors.

    Args:
        Eigenvalues (eigval): np.array
        Explained variance (explvar): np.array
        savefig: bool

    Returns:
        screen plot and if savefig a .png
    """
    fig, ax = plt.subplots()  # Default size of the figure
    
    # Individual eigenvector variance
    ax.bar(range(len(eigval)), eigval / np.sum(eigval), alpha=0.5,
           align='center', color='#004e92', label='Individual explained variance')
    
    # Cumulative variance
    csum_explvar = np.cumsum(explvar)
    ax.step(range(len(csum_explvar)), csum_explvar, where='mid',
            color='#8b2a44', label='Cumulative explained variance')

    # Some plot parameters
    ax.set_xlabel('Principal component index', fontsize=14)
    ax.set_ylabel('Explained variance ratio', fontsize=14)
    ax.legend(fontsize=14, loc='center right')

    # ax.set_xticks(np.arange(0, len(eigval)))
    # ax.set_xticklabels(np.arange(1, len(eigval) + 1))
    even_indices = np.arange(1, len(eigval), 2)
    even_labels = even_indices + 1
    ax.set_xticks(even_indices)
    ax.set_xticklabels(even_labels)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    ax.tick_params(axis='both', which='both', direction='in', labelsize=14, 
                   top=True, bottom=True, left=True, right=True)

    ax.set_xlim([-0.75, len(eigval)])

    plt.tight_layout()

    if savefig:
        plt.savefig('explained_var.png', dpi=300)
    else:
        pass
    
    plt.show()


def pcs_composition(eigvec: np.array, savefig: bool=False):
    """Plots the PCs and the NMs to have a picture about the PCs composition.

    Args:
        Eigenvectors (eigvec): np.array
        savefig: bool

    Returns:
        screen plot and if savefig a .png
    """
    # fig, ax = plt.subplots(figsize=(10,6))
    fig, ax = plt.subplots()  # Default figsize

    cax = ax.imshow(np.abs(eigvec**2).T, cmap='cividis')

    ax.set_xlabel("Principal component index", fontsize=14)
    ax.set_ylabel("Normal Mode index", fontsize=14)

    """
    ax.set_xticks(list(range(1,25,2)))  # Tick positions
    ax.set_xticklabels(list(range(2,25,2)))  # Tick positions
    ax.set_yticks(list(range(1,25,2)))  # Tick positions
    ax.set_yticklabels(list(range(2,25,2)))  # Tick positions

    ax.set_yticks(list(range(0,24))) 
    ax.set_yticklabels([r"CH$_3$ asym. torsion",
                        r"CH$_3$  sym. torsion",
                        r"NN torsion",
                        r"CNN asym. bend",
                        r"CNN  sym. bend",
                        r"CH$_3$ rock",
                        r"CN asym. stretch",
                        r"CH$_3$ rock",
                        r"CH$_3$ rock",
                        r"CH$_3$ rock",
                        r"CN sym. stretch",
                        r"CH$_3$ asym. bend",
                        r"CH$_3$  sym. bend",
                        r"CH$_3$ asym. bend",
                        r"CH$_3$ asym. bend",
                        r"CH$_3$  sym. bend",
                        r"CH$_3$ asym. bend",
                        r"NN stretch",
                        r"CH$_3$  sym. stretch",
                        r"CH$_3$  sym. stretch",
                        r"CH$_3$ asym. stretch",
                        r"CH$_3$ asym. stretch",
                        r"CH$_3$ asym. stretch",
                        r"CH$_3$ asym. stretch"])
    """

    ax.tick_params(axis='x', labelsize=14)  # X-axis ticks
    ax.tick_params(axis='y', labelsize=14)  # Y-axis ticks

    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel("PC coefficient squared", fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    
    if savefig:
        plt.savefig("PCs2_NMs_composition.png", dpi=300)
        # plt.savefig("PCs2_nms_trans-AZM_square.png", dpi=300)
    else:
        pass
    
    plt.show()

    return None




if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Analyze a PCA pickle file.\n\
                                     Plot the explained variance as a function of the eigenvectors number.\n\
                                     Filter the trajectory along the selected PC.")
    parser.add_argument("-f", "--file", required=True, help="Path to the PCA .pickle file")
    
    # Arguments related to the plot of the explained variance
    parser.add_argument("--plot", action="store_true", help="Plot explained variance")
    parser.add_argument("--savefig", action="store_true", help="Save a .png of the explained variance plot")
    
    # Arguments related to the plot of the PCs squared
    parser.add_argument("--plot_pc2", action="store_true", help="Plot principal components squared")

    args = parser.parse_args()
    filename = args.file
    
    plot = args.plot
    plot_pc2 = args.plot_pc2
    savefig = args.savefig

    # Check dependencies: --savefig requires --plot
    if args.savefig and not (args.plot or args.plot_pc2):
        parser.error("--savefig requires --plot and/or --plot_pc2. Use --plot --savefig")

    # Loading the PCA pickle file
    pca_pickle = pickle_load(filename)

    # Eigenvectors
    eigvec = pca_pickle.components_
    # Eigenvalues
    eigval = pca_pickle.explained_variance_
    # Explained variance (eigenvalues / sum(eigenvalues))
    explvar = pca_pickle.explained_variance_ratio_

    if plot:
        plot_explained_var(eigval, explvar, savefig)
        if savefig:
            print("Plot saved to: explained_var.png")
        else:
            print("Plot not saved. Set --savefig if you want to save a .png.")
    else:
        print("Plot not shown. Enable --plot if you want to plot.")

    if plot_pc2:
        pcs_composition(eigvec, savefig)
