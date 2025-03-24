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
The script reduces the generated initial consitions

  Vincent Delmas (2023)

*******************************************************************************
"""

import glob
import os
import shutil
from typing import Tuple, List
import numpy as np
import warnings
import pickle


""" --- PARAMETERS ------------------------------------------------------------------------------------------- """

warnings.filterwarnings("ignore", category=UserWarning)

# From [Bohr] to [Angstrom]
BOHR_TO_ANG = 0.529177249

# From [g]/[mol] to [amu]
U_TO_AMU = 1. / 5.4857990943e-4

# Atomic Weights common isotopes - From https://chemistry.sciences.ncsu.edu/msf/pdf/IsotopicMass_NaturalAbundance.pdf
MASSES = {'H': 1.007825 * U_TO_AMU,
          'He': 4.002603 * U_TO_AMU,
          'Li': 7.016004 * U_TO_AMU,
          'Be': 9.012182 * U_TO_AMU,
          'B': 11.009305 * U_TO_AMU,
          'C': 12.000000 * U_TO_AMU,
          'N': 14.003074 * U_TO_AMU,
          'O': 15.994915 * U_TO_AMU,
          'F': 18.998403 * U_TO_AMU,
          'Ne': 19.992440 * U_TO_AMU,
          'Na': 22.989770 * U_TO_AMU,
          'Mg': 23.985042 * U_TO_AMU,
          'Al': 26.981538 * U_TO_AMU,
          'Si': 27.976927 * U_TO_AMU,
          'P': 30.973762 * U_TO_AMU,
          'S': 31.972071 * U_TO_AMU,
          'Cl': 34.968853 * U_TO_AMU,
          'Ar': 39.962383 * U_TO_AMU,
          'K': 38.963707 * U_TO_AMU,
          'Ca': 39.962591 * U_TO_AMU,
          'Sc': 44.955910 * U_TO_AMU,
          'Ti': 47.947947 * U_TO_AMU,
          'V': 50.943964 * U_TO_AMU,
          'Cr': 51.940512 * U_TO_AMU,
          'Mn': 54.938050 * U_TO_AMU,
          'Fe': 55.934942 * U_TO_AMU,
          'Co': 58.933200 * U_TO_AMU,
          'Ni': 57.935348 * U_TO_AMU,
          'Cu': 62.929601 * U_TO_AMU,
          'Zn': 63.929147 * U_TO_AMU,
          'Ga': 68.925581 * U_TO_AMU,
          'Ge': 73.921178 * U_TO_AMU,
          'As': 74.921596 * U_TO_AMU,
          'Se': 79.916522 * U_TO_AMU,
          'Br': 78.918338 * U_TO_AMU,
          'Kr': 83.911507 * U_TO_AMU,
          'Rb': 84.911789 * U_TO_AMU,
          'Sr': 87.905614 * U_TO_AMU,
          'Y': 88.905848 * U_TO_AMU,
          'Zr': 89.904704 * U_TO_AMU,
          'Nb': 92.906378 * U_TO_AMU,
          'Mo': 97.905408 * U_TO_AMU,
          'Tc': 98.907216 * U_TO_AMU,
          'Ru': 101.904350 * U_TO_AMU,
          'Rh': 102.905504 * U_TO_AMU,
          'Pd': 105.903483 * U_TO_AMU,
          'Ag': 106.905093 * U_TO_AMU,
          'Cd': 113.903358 * U_TO_AMU,
          'In': 114.903878 * U_TO_AMU,
          'Sn': 119.902197 * U_TO_AMU,
          'Sb': 120.903818 * U_TO_AMU,
          'Te': 129.906223 * U_TO_AMU,
          'I': 126.904468 * U_TO_AMU,
          'Xe': 131.904154 * U_TO_AMU,
          'Cs': 132.905447 * U_TO_AMU,
          'Ba': 137.905241 * U_TO_AMU,
          'La': 138.906348 * U_TO_AMU,
          'Ce': 139.905435 * U_TO_AMU,
          'Pr': 140.907648 * U_TO_AMU,
          'Nd': 141.907719 * U_TO_AMU,
          'Pm': 144.912744 * U_TO_AMU,
          'Sm': 151.919729 * U_TO_AMU,
          'Eu': 152.921227 * U_TO_AMU,
          'Gd': 157.924101 * U_TO_AMU,
          'Tb': 158.925343 * U_TO_AMU,
          'Dy': 163.929171 * U_TO_AMU,
          'Ho': 164.930319 * U_TO_AMU,
          'Er': 165.930290 * U_TO_AMU,
          'Tm': 168.934211 * U_TO_AMU,
          'Yb': 173.938858 * U_TO_AMU,
          'Lu': 174.940768 * U_TO_AMU,
          'Hf': 179.946549 * U_TO_AMU,
          'Ta': 180.947996 * U_TO_AMU,
          'W': 183.950933 * U_TO_AMU,
          'Re': 186.955751 * U_TO_AMU,
          'Os': 191.961479 * U_TO_AMU,
          'Ir': 192.962924 * U_TO_AMU,
          'Pt': 194.964774 * U_TO_AMU,
          'Au': 196.966552 * U_TO_AMU,
          'Hg': 201.970626 * U_TO_AMU,
          'Tl': 204.974412 * U_TO_AMU,
          'Pb': 207.976636 * U_TO_AMU,
          'Bi': 208.980383 * U_TO_AMU,
          'Po': 208.982416 * U_TO_AMU,
          'At': 209.987131 * U_TO_AMU,
          'Rn': 222.017570 * U_TO_AMU,
          'Fr': 223.019731 * U_TO_AMU,
          'Ra': 226.025403 * U_TO_AMU,
          'Ac': 227.027747 * U_TO_AMU,
          'Th': 232.038050 * U_TO_AMU,
          'Pa': 231.035879 * U_TO_AMU,
          'U': 238.050783 * U_TO_AMU,
          'Np': 237.048167 * U_TO_AMU,
          'Pu': 244.064198 * U_TO_AMU,
          'Am': 243.061373 * U_TO_AMU,
          'Cm': 247.070347 * U_TO_AMU,
          'Bk': 247.070299 * U_TO_AMU,
          'Cf': 251.079580 * U_TO_AMU,
          'Es': 252.082972 * U_TO_AMU,
          'Fm': 257.095099 * U_TO_AMU,
          'Md': 258.098425 * U_TO_AMU,
          'No': 259.101024 * U_TO_AMU,
          'Lr': 262.109692 * U_TO_AMU,
          'Rf': 267. * U_TO_AMU,
          'Db': 268. * U_TO_AMU,
          'Sg': 269. * U_TO_AMU,
          'Bh': 270. * U_TO_AMU,
          'Hs': 270. * U_TO_AMU,
          'Mt': 278. * U_TO_AMU,
          'Ds': 281. * U_TO_AMU,
          'Rg': 282. * U_TO_AMU,
          'Cn': 285. * U_TO_AMU,
          'Nh': 286. * U_TO_AMU,
          'Fl': 289. * U_TO_AMU,
          'Mc': 290. * U_TO_AMU,
          'Lv': 293. * U_TO_AMU,
          'Ts': 294. * U_TO_AMU,
          'Og': 294. * U_TO_AMU
          }

""" ---------------------------------------------------------------------------------------------------------- """


def read_xyz_file(filename: str) -> Tuple[List, List]:
    """Reads an .xyz file and extracts atomic coordinates and symbols.

    Args:
        filename (str): The path to the .xyz file to read.

    Returns:
        Tuple[List, List]: A tuple containing two lists:
            - The first list contains coordinates as lists of floats.
            - The second list contains the atomic symbols corresponding to each coordinate.
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

    return coordinates, symbols


def read_red_file(filename: str) -> List:
    """Reads a file and extracts atomic coordinates.

    Args:
        filename (str): The path to the file to read.

    Returns:
        List: A list of coordinates as lists of floats.
    """
    coordinates = []
    with open(filename) as file:
        next(file)
        for line in file:
            if line.strip():
                x, y, z = line.split()
                coordinates.append([float(x), float(y), float(z)])

    return coordinates


def create_xyz_file(filepath: str, coordinates: List, symbols: List):
    """Writes atomic coordinates and symbols to a new .xyz file.

    Args:
        filepath (str): The path where the .xyz file will be saved.
        coordinates (List): A list of coordinates, where each coordinate is a list of floats.
        symbols (List): A list of atomic symbols corresponding to each set of coordinates.
    """
    with open(filepath, "w") as file:
        file.write(str(len(coordinates)) + "\n")
        file.write("\n")

        for symbol, coord in zip(symbols, coordinates):
            line = symbol + "\t" + "\t".join(str(x) for x in coord) + "\n"
            file.write(line)


def delete_file(file_path: str):
    """Deletes a file at the specified path.

    Args:
        file_path (str): The path to the file to be deleted.

    Raises:
        Exception: If any error occurs during the deletion.
    """
    try:
        os.remove(file_path)
    except FileNotFoundError:
        # Ignore if the file does not exist
        pass
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")


def pickle_load(fname: str):
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


def reduce_geoms(pickle_file: str):
    """Reduces the geometries named `geom_*.xyz` present in the `TRAJ*/` folders.

    Args:
        pickle_file (str): location of the pickle binary file.

    Returns:
        None. The function writes the `red_geom_*.xyz` inside each `TRAJ*/` folder.
    """
    all_geoms = glob.glob("TRAJ*/geom_*.xyz")
    for geom in all_geoms:
        coord, symbols = read_xyz_file(geom)

        # Change form Angstroms to Bohr
        coord = np.asarray(coord)
        coord *= 1/BOHR_TO_ANG

        # Create paths
        red_in = os.path.join(os.path.dirname(geom), "red.in")
        red_out = os.path.join(os.path.dirname(geom), "red.out")

        # Create <red.in> file
        with open(red_in, "w") as file:
            file.write("[DATA]\n")
            for c in coord:
                file.write(" ".join(str(x) for x in c) + "\n")
            file.write("[MASS]\n")
            for val in masses:
                file.write(str(val) + "\n")
            file.write("[GEOM]\n")
            for c in coord:
                file.write(" ".join(str(x) for x in c) + "\n")

        # Transform data
        os.system(f"python transformer.py {red_in} red.out pca geom {pickle_file}")

        # Change location of <red.out> file
        shutil.move("red.out", red_out)

        # New coords
        new_coord = read_red_file(red_out)

        # Change from Bohr to [Ang]
        new_coord = np.asarray(new_coord)
        new_coord *= BOHR_TO_ANG

        # New xyz file, red_geom_*.xyz
        create_xyz_file(geom.replace("geom_", "red_geom_"), new_coord, symbols)

        # Delete <red.in>, <red.out> and <coms_data.json>
        geom_directory = os.path.dirname(geom)
        delete_file(os.path.join(geom_directory, "red.in"))
        delete_file(os.path.join(geom_directory, "red.out"))
        delete_file("coms_data.json")


def reduce_velocities(pickle_file):
    """Reduces the velocities named `velocity_*.xyz` present in the `TRAJ*/` folders.

    Args:
        pickle_file (str): location of the pickle binary file.

    Returns:
        None. The function writes the `red_velocity_*.xyz` inside each `TRAJ*/` folder.
    """
    all_velocities = glob.glob("TRAJ*/velocity_*.xyz")
    for vel in all_velocities:
        coord = np.loadtxt(vel)

        # Create paths
        red_in = os.path.join(os.path.dirname(vel), "red.in")
        red_out = os.path.join(os.path.dirname(vel), "red.out")

        # Create <red.in> file
        with open(red_in, "w") as file:
            file.write("[DATA]\n")
            for c in coord:
                file.write(" ".join(str(x) for x in c) + "\n")
            file.write("[MASS]\n")
            for val in masses:
                file.write(str(val) + "\n")
            # The following is not the [GEOM] but the velocity
            file.write("[GEOM]\n") 
            for c in coord:
                file.write(" ".join(str(x) for x in c) + "\n")

        # Transform data
        os.system(f"python transformer.py {red_in} red.out pca vel {pickle_file}")

        # Change location of <red.out> file
        shutil.move("red.out", red_out)

        # New coords
        new_coord = np.asarray(read_red_file(red_out))

        # New velocity file
        np.savetxt(vel.replace("velocity_", "red_velocity_"), new_coord)

        # Delete <red.in>, <red.out> and <coms_data.json>
        vel_directory = os.path.dirname(vel)
        delete_file(os.path.join(vel_directory, "red.in"))
        delete_file(os.path.join(vel_directory, "red.out"))
        delete_file("coms_data.json")


if __name__ == "__main__":
    pickle_file = glob.glob("*.pickle")[0]
    
    # Extracting atom names to avoid the manual
    # poplation of the MASS array that depends on the studied molecule.
    pickle = pickle_load(f"{pickle_file}")
    atom_names = pickle.atom_names_list
    masses = [MASSES[atom_names[i]] for i in range(len(atom_names))]

    # Reduce geoms
    reduce_geoms(pickle_file)

    # reduce velocities
    reduce_velocities(pickle_file)
