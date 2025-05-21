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
## Create PCA containers of reduced dimensions for molecular dynamics

## Overview
This script performs dimensionality reduction on molecular dynamics
`ensemble` using PCA. The code uses custom classes from the `scripts` module,
specifically `TSH.FitterReducer` for the reduction and PCA and the
`TSH.NormalModes` to creates the normal modes data using an OpenMolcas output
freqency file. The data are loaded from precomputed ensembles using the
`pickle_load` function from `scripts.utilities`.

## Instructions
1. Ensure the necessary data files are available: the `ensemble.pickle` and the
   frequency output file from OpenMolcas.
2. Adjust the file paths in the script to point to the appropriate data files
   for your specific molecule.
3. Set the desired dimensions (`dim`) for the PCA in the for loop.

## Requirements
- `ensemble.pickle`: Molecular ensemble data created from full dimensions
  simulations.
- `.output`: Frequency data obtained from OpenMolcas calculations.
  
  Vincent Delmas (2023)

*******************************************************************************
"""

import sys
import os
import argparse

""" ----------------------------------------------------------------------------------------------------- """

# Add the /src directory to $PATH
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import scripts.class_TSH as TSH
from scripts.utilities import pickle_load

""" ----------------------------------------------------------------------------------------------------- """


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='create_ensemble.py',
        description='Creates an ensemble of all trajectories present in the current directory',
        epilog='Don\'t forget to drink water!'
    )

    parser.add_argument('-i', '--input', type=str, default='ensemble.pickle',
                        help='name of the input file (default: ensemble.pickle)')
    parser.add_argument('-f', '--freq_file', type=str, default='freq.output',
                        help='name of the frequency file (default: freq.output)')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # Full dimensions ensembles used to create PCA containers
    ensemble = pickle_load(args.input)

    # Create object containing <cart2nm>, <nm2cart>, and <nm_matrix> matrices
    # TODO: add a functionality that distinguish between only OPT and OPT+FREQ files
    # because if you put below an .output file that is an OPT followed by MCKINLEY
    # it will take the very initial geometry and the normal modes of the OPT geometry
    nm = TSH.NormalModes(filename=args.freq_file)

    print("Enter the number of principal components to keep")
    print("You can specify multiple integers to generate multiple files")
    print("example: 2 4 31\n")

    while True:
        user_input = input('> ')
        try:
            answer = [int(x) for x in user_input.split()]

            if True in [1 > x > nm.nb_nm - 1 for x in answer]:
                raise ValueError
            else:
                break

        except ValueError:
            print("Input only integers lower than the number of normal modes\n")
            continue

    # Create PCA object
    fr = TSH.FitterReducer(ensemble)

    # Apply a transformation
    fr.featurizer(repr="nm", nm=nm, kabsch=True, COM=True, mass_w=False)

    # Create containers with N dimensions
    for dim in answer:

        # Apply PCA and save as pickle file
        fr.apply_pca(n_comp=dim)
