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
This file contains all the classes used for post-treatment as well as creating
Ensemble of trajectories, Normal modes, Fit and Reduce geometries from full
dimensions dynamics.

  Morgane Vacher, Vincent Delmas, Bartosz Ciborowski,
  Alessandro Nicola Nardi (2025)

*******************************************************************************
"""

from __future__ import annotations
import itertools
import os
import re
import sys
from math import ceil
from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from scripts.constants import *
from scripts.utilities import get_idx, get_col_array, gs, kabsch_algorithm, angle, dihedral, pickle_save


""" ---------------------------------------------------------------------------------------------------------- """

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

VALID_PROGRAMS = ["molcas", "sharc", "nx"]

# Energy conservation threshold [eV]
ETOT_THRESH = 0.5

""" ---------------------------------------------------------------------------------------------------------- """


class FitterReducer:
    """Take an ensemble of trajectories and create featurization of the data and reduce
    its dimensionality using different algorithms. Saved as pickle file. """

    def __init__(self, ensemble: Ensemble):
        """Initialize FitterReducer with an ensemble.

        Args:
            ensemble (Ensemble): An Ensemble object containing trajectories.
        """
        self.ens = ensemble
        self.featurized_data_ = None
        self.repr = ""
        self.nm = None

    def _check_featurized(self):
        """Check if data has been featurized.

        Raises:
            Exception: If featurized_data_ is None, indicating data has not been featurized yet.
        """
        if self.featurized_data_ is None:
            raise Exception("You first need to featurize the data using the featurizer method. Exiting.")

    def featurizer(self, repr: str = "cart", kabsch: bool = True, COM: bool = False, nm: NormalModes = None,
                   mass_w: bool = False):
        """Featurize the data into different representations.

        Available representations are:
            - 'cart': Cartesian coordinates.
            - 'force': Forces.
            - 'nm': Normal modes coordinates.

        Args:
            repr (str, optional): Representation type ('cart', 'force', 'nm'). Defaults to "cart".
            kabsch (bool, optional): Apply Kabsch algorithm for alignment. Defaults to True.
            COM (bool, optional): Use Center of Mass for Kabsch alignment. Defaults to False.
            nm (NormalModes, optional): NormalModes object for 'nm' representation. Defaults to None.
            mass_w (bool, optional): Mass weight the data. Defaults to False.

        Raises:
            Exception: If incompatible arguments are provided or representation is not recognized.
        """
        # Check incompatible arguments
        if COM is True and kabsch is False:
            raise Exception("Can't use the kabsch with COM without kabsch argument set to True.")
        if (repr == "force" and kabsch is True) or (repr == "force" and COM is True):
            raise Exception("Can't kabsch the forces.")
        if repr == "nm" and nm is None:
            raise Exception("A <TSH.NormalModes> instance is necessary for normal modes representation.")
        if repr == "nm" and mass_w is True:
            raise Exception("Can't mass Weight the data and get the normal mode representation.")

        # Setup representation attributes
        self.repr = repr
        self.nm = nm

        # Get all data from representation
        if repr == "force":
            data_ens = self.ens.get_all_forces().values()
        elif repr == "cart" or repr == "nm":
            data_ens = self.ens.get_all_geometries().values()
        else:
            raise Exception(f"The '{repr}' repr pattern not recognized. Values allowed: 'force', 'cart' or 'nm'.")

        # Create masses and sqrt_masses
        masses = np.asarray([ELEMENTS_ATOMIC_MASS[a] * AMU_TO_ME for a in self.ens.trajs[0].frames[0].atoms])
        sqrt_masses = np.sqrt(masses)

        # Get representation
        featurized_data = []
        for traj in data_ens:
            ref_data = traj[0]
            for data in traj:
                if kabsch:
                    data = kabsch_algorithm(data, ref_data, weights=masses if COM else None)
                if mass_w:
                    data = (data.reshape(len(sqrt_masses), 3) * sqrt_masses.reshape(-1, 1)).ravel()
                if repr == "nm":
                    # Angstroms to Bohr conversion
                    data *= ANG_TO_BOHR
                    data = nm.from_cart2nm(data)

                featurized_data.append(data)

        # The featurized data are converted as array and populate the attribute
        self.featurized_data_ = np.asarray(featurized_data)

    def apply_pca(self, var: float = 90., n_comp: int = None, **kwargs):
        """Apply Principal Component Analysis (PCA) on the featurized data.

        If n_comp is not provided, it finds the minimum number of components to explain at least 'var' variance.

        Args:
            var (float, optional): Minimum variance percentage to be explained. Defaults to 90.0.
            n_comp (int, optional): Number of principal components to use. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to sklearn.decomposition.PCA.
        """
        self._check_featurized()

        if n_comp:
            pca = PCA(n_components=n_comp, svd_solver="full", **kwargs)
            pca.fit(self.featurized_data_)
            print(f"\n*** Variance = {sum(pca.explained_variance_ratio_) * 100:4f} %\n")
        else:
            for i in range(1, self.featurized_data_[0].shape[0]):
                pca = PCA(n_components=i, svd_solver="full", **kwargs)
                pca.fit(self.featurized_data_)

                variance = sum(pca.explained_variance_ratio_) * 100
                n_comp = i
                if variance >= var:
                    print(f"{i} DIM explains at least {var}% of the variance. Stopping.")
                    break

        # Add representation (repr) to PCA object from sklearn.decomposition
        # noinspection PyUnboundLocalVariable
        pca.repr = self.repr
        # Saving a List of atom names
        pca.atom_names_list = self.ens.trajs[0].frames[0].atoms

        if self.repr == "nm":
            """Get the necessary data from the NormalModes class to save with the PCA pickle. """

            # noinspection PyUnboundLocalVariable
            pca.cart2nm = self.nm.cart2nm
            pca.nm2cart = self.nm.nm2cart
            pca.geom_ref_bohr = self.nm.geom_ref_bohr
            # Saving a List of atom names
            pca.atom_names_list = self.ens.trajs[0].frames[0].atoms

        # Save PCA object to pickle
        pickle_save(f"PCA_{n_comp}_comp_{self.repr}.pickle", pca)

    def save_as_csv(self, name: str = "data_saved"):
        """Save the featurized data to a CSV file.

        Args:
            name (str, optional): Base name for the CSV file. Defaults to "data_saved".
        """
        self._check_featurized()

        df = pd.DataFrame(self.featurized_data_)
        print(df.head(5))
        df.to_csv(f"{name}.csv", header=False, index=False)


class NormalModes:
    """Take a Gaussian or Molcas frequency calculation output file, and create the matrices to convert from
    cartesian [Bohr] to mass-frequency scaled normal modes (as used in vMCG). """

    def __init__(self, filename: str, filetype: str = "mknly"):
        """Initialize NormalModes object.

        Args:
            filename (str): Path to the Gaussian or Molcas frequency calculation output file.
            filetype (str): Type of the job in the frequency calculation output file (freq, or opt+freq).
                            It is needed only for OpenMolcas to correctly take the geometry of the NMs.
                            Default is "mknly", which corresponds to only frequencies calculation.
        """
        self.filetype = filetype
        self.nb_atoms = None
        self.nb_cart = None
        self.nb_nm = None
        self.symbols = []
        self.masses = []

        self.geom_ref_bohr = []
        self.geom_ref_ang = []

        # Open file and save as list of str
        try:
            with open(filename) as fp:
                self.data = fp.read().splitlines()
        except FileNotFoundError as e:
            raise FileNotFoundError("Make sure to specify the name of the frequency file using the '-f' flag") from e

        # Get <Gaussian> or <Molcas> parser
        if len(get_idx(self.data, "Entering Gaussian System")) != 0:
            freqs, normal_modes = self._parse_Gaussian()
        elif len(get_idx(self.data, "This copy of MOLCAS")) != 0 or len(get_idx(self.data, "This run of MOLCAS")) != 0:
            freqs, normal_modes = self._parse_Molcas()
        elif len(get_idx(self.data, "[Molden Format]")) != 0:
            freqs, normal_modes = self._parse_Molden()
        else:
            raise Exception("Only <Gaussian>, <Molcas> and <Molden> frequency files are supported for now.")

        # Convert frequencies to Hartree
        freq_hartree = abs(freqs * INV_CM_TO_HARTREE)

        # Mass scale normal modes
        nm_mass_scaled = normal_modes * np.sqrt(self.masses)

        # Gram-Schmidt ortho-normalize mass-scaled normal modes
        self.nm_matrix = gs(nm_mass_scaled)

        # Dimensionless normal modes -> scale by sqrt(mass * freq) - NOT ORTHOGONAL
        self.cart2nm = ((self.nm_matrix * (np.sqrt(self.masses * AMU_TO_ME))).T * np.sqrt(freq_hartree)).T

        # Matrix to go from normal modes to cartesian - NOT ORTHOGONAL
        self.nm2cart = (self.nm_matrix / (np.sqrt(self.masses * AMU_TO_ME))).T / np.sqrt(freq_hartree)

    def _parse_Gaussian(self):
        """Parse Gaussian output data to extract frequencies and normal modes.

        This function extracts the following information from the Gaussian output data
        stored in `self.data`:

        - Atomic symbols: Extracted from the "Standard orientation" section to deduce atomic masses.
        - Atomic masses:  Determined based on the atomic symbols using `ELEMENTS_ATOMIC_MASS`.
        - Geometry:       Extracted from the "Standard orientation" section in Angstroms and converted to Bohr.
        - Frequencies:    Extracted from the "Frequencies" section in cm-1.
        - Normal modes:   Extracted from the "Frequencies" section, corresponding to the frequencies.

        Returns:
            tuple: A tuple containing:
                - freqs (np.ndarray): Array of frequencies in cm-1.
                - normal_modes (np.ndarray): Array of normal modes.
                                            Shape is (nb_nm, nb_cart), where nb_nm is the number of normal modes
                                            and nb_cart is the number of Cartesian coordinates.
        """
        # Get atomic symbols to deduce masses [AMU]
        idx_geom = get_idx(self.data, "Standard orientation:")[0] + 5

        # Get number of atoms
        idx_atoms = get_idx(self.data, 'NAtoms')[0]
        self.nb_atoms = int(self.data[idx_atoms].split()[1])
        self.nb_cart = self.nb_atoms * 3

        # Round to nearest integer (in case of float representation)
        atomic_numbers = get_col_array(self.data[idx_geom: idx_geom + self.nb_atoms], 1).round().astype(int)
        self.symbols = [ATOMIC_NUMBER_TO_SYMBOL.get(num, "?") for num in atomic_numbers]

        # Add atomic Mass using symbols
        masses = []
        for symbol in self.symbols:
            masses.extend([ELEMENTS_ATOMIC_MASS.get(symbol)]*3)
        self.masses = np.asarray(masses)
        
        # Get origin geometry [Angstroms]
        geom = np.zeros(self.nb_cart)
        for i in range(self.nb_atoms):
            geom[i*3: i*3+3] = [float(c) for c in self.data[idx_geom + i].split()[3: 6]]

        # Geom [Ang] and [Bohr]
        self.geom_ref_ang = geom
        self.geom_ref_bohr = geom * ANG_TO_BOHR

        # Get number of normal modes
        idx_nm = get_idx(self.data, "Frequencies --")
        nb_nm = 0
        for idx in idx_nm:
            nb_nm += len(self.data[idx].split()) - 2
        self.nb_nm = nb_nm

        # Read frequencies [cm-1] and normal modes from output file
        freqs = np.zeros(self.nb_nm)
        normal_modes = np.zeros((self.nb_nm, self.nb_cart))

        # Normal modes are printed 3 by 3
        idx_nm = idx_nm[0]
        for i in range(self.nb_nm//3):
            freqs[i*3: i*3+3] = [float(f) for f in self.data[idx_nm].split()[2:]]
            idx_nm += 5
            
            # `list3` contains the NumPy arrays of the x, y, z columns of the parsed NMs
            list3 = [get_col_array(self.data[idx_nm: idx_nm + self.nb_atoms], 2+n) for n in range(9)]
            # `grouped` contains the lists of the NumPy arrays of the x, y, z of the NMs 
            grouped = [list3[i:i+3] for i in range(0, len(list3), 3)]
            # Initialising the array containing the x, y, z of the three parsed NMs 
            array3 = np.zeros((3, self.nb_cart))
            for row_idx, (x, y, z) in enumerate(grouped):
                # Interleave the x, y, z values for each atom
                combined = np.column_stack([x, y, z]).flatten()
                array3[row_idx] = combined

            normal_modes[i*3: i*3+3] = array3
            idx_nm += 2 + self.nb_atoms

        return freqs, normal_modes

    def _parse_Molcas(self):
        """Parse Molcas output data to extract frequencies and normal modes.

        This function extracts the following information from the OpenMolcas output data
        stored in `self.data`:

        - Atomic symbols: Extracted from the "Cartesian Coordinates" section to deduce atomic masses.
        - Atomic masses:  Determined based on the atomic symbols using `ELEMENTS_ATOMIC_MASS`.
        - Geometry:       Extracted from the "Cartesian Coordinates" section in Bohr and converted to Angstroms.
        - Frequencies:    Extracted from the "Frequencies" section in cm-1.
        - Normal modes:   Extracted from the "Frequencies" section, corresponding to the frequencies.

        Returns:
            tuple: A tuple containing:
                - freqs (np.ndarray): Array of frequencies in cm-1.
                - normal_modes (np.ndarray): Array of normal modes.
                                            Shape is (nb_nm, nb_cart), where nb_nm is the number of normal modes
                                            and nb_cart is the number of Cartesian coordinates.
        """
        # Detecting if the output file contains frequency or optimisation steps and frequencies at the end 
        if self.filetype == "mckly":
            # Get symbols and geometry [Angstroms] at the start of the file
            idx_geom = get_idx(self.data, "Cartesian Coordinates")[0] + 4
            # Get number of atoms
            nb_atoms = 0
            for line in self.data[idx_geom:]:
                if line.strip() == '':
                    self.nb_atoms = nb_atoms
                    self.nb_cart = self.nb_atoms * 3
                    break
                nb_atoms += 1

            print(f"Found {self.nb_atoms} atoms in the OpenMolcas frequency files.")
            geom_data = [get_col_array(self.data[idx_geom: idx_geom + self.nb_atoms], n+1) for n in range(4)]
        elif self.filetype == "optmck":
            # Get symbols and geometry [Angstroms] right before MCKINLEY
            idx_geom = get_idx(self.data, "Nuclear coordinates of the final structure / Bohr")[0] + 3
            # Get number of atoms
            nb_atoms = 0
            for line in self.data[idx_geom:]:
                if line.strip() == '':
                    self.nb_atoms = nb_atoms
                    self.nb_cart = self.nb_atoms * 3
                    break
                nb_atoms += 1

            print(f"Found {self.nb_atoms} atoms in the OpenMolcas frequency files.")
            geom_data = [get_col_array(self.data[idx_geom: idx_geom + self.nb_atoms], n) for n in range(4)]
        else:
            raise Exception("""
            *** Please specify the type of the <OpenMolcas> output file. ***

            Options:
                [mckly]   for a frequency output file
                [optmck]  for an optimisation and frequency calculation
            """)

        # Get atomic symbols to deduce masses [AMU]
        self.symbols = [re.sub(r"\d+", "", s) for s in geom_data[0]]

        # Get ref geometry
        self.geom_ref_bohr = np.asarray(list(zip(*geom_data[1:]))).ravel()
        self.geom_ref_ang = self.geom_ref_bohr * BOHR_TO_ANG

        # Add atomic Mass using symbols
        masses = []
        for symbol in self.symbols:
            masses.extend([ELEMENTS_ATOMIC_MASS.get(symbol)]*3)
        self.masses = np.asarray(masses)

        # Read frequencies [cm-1] and normal modes (printed 6 by 6) from output file
        nm_indices = get_idx(self.data, "Frequency:")
        limit_index = get_idx(self.data, 'Principal components of the normal modes')[0]
        nm_indices = [nm_idx for nm_idx in nm_indices if nm_idx < limit_index]

        nb_nm = 0
        freqs, normal_modes = [], []
        for nm, idx in enumerate(nm_indices):
            # TODO what should be done about imaginary frequencies?
            freqs.extend([(-float(f[1:]) if f[0] == "i" else float(f)) for f in self.data[idx].split()[1:]])
            num_columns = len(self.data[idx].split()[1:])
            nb_nm += num_columns
            normal_modes.extend([get_col_array(self.data[idx+5: idx+5+self.nb_cart], 2+n) for n in range(num_columns)])

        self.nb_nm = nb_nm
        print(f"Found {self.nb_nm} normal modes in the OpenMolcas frequency files")

        freqs = np.asarray(freqs)
        normal_modes = np.asarray(normal_modes).reshape((self.nb_nm, self.nb_cart))

        return freqs, normal_modes

    def _parse_Molden(self):
        """Parses the Molden-format frequency file to extract vibrational data.
        This method assumes that the .molden file is generated with the Molden program.
        The molden files generated with other programs may differ in one or more headers. 

        This method reads:
        - the number of atoms and cartesian degrees of freedom,
        - the atomic symbols and geometry in Bohr and Angstrom units,
        - the atomic masses based on symbols,
        - vibrational frequencies,
        - and the corresponding normal modes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                A tuple containing:
                    - `freqs` (np.ndarray): 1D array of vibrational frequencies.
                    - `normal_modes` (np.ndarray): 2D array of normal mode vectors 
                    with shape (nb_nm, nb_cart), where `nb_nm` is the number of 
                    vibrational modes and `nb_cart` is the number of Cartesian coordinates.
        """
        # Reading the number of atoms
        idx_geom = get_idx(self.data, "[GEOMETRIES]")[0] + 1
        self.nb_atoms = int(self.data[idx_geom])
        print(f"Found {self.nb_atoms} atoms in the Molden frequency file.")
        self.nb_cart = self.nb_atoms * 3

        # Reading the geometry
        idx_geom += 2  # increasing the index by 2 to go from number of atoms to the first row of the coordinates
        geom_data = [get_col_array(self.data[idx_geom: idx_geom + self.nb_atoms], n) for n in range(4)]

        # Get atomic symbols to deduce masses [AMU]
        self.symbols = [re.sub(r"\d+", "", s) for s in geom_data[0]]

        # Get ref geometry
        self.geom_ref_bohr = np.asarray(list(zip(*geom_data[1:]))).ravel() * ANG_TO_BOHR
        self.geom_ref_ang = self.geom_ref_bohr * BOHR_TO_ANG

        # Add atomic Mass using symbols
        masses = []
        for symbol in self.symbols:
            masses.extend([ELEMENTS_ATOMIC_MASS.get(symbol)]*3)
        self.masses = np.asarray(masses)

        # Reading frequencies and normal modes
        idx_lastvib = get_idx(self.data, "vibration")[-1]
        self.nb_nm = int(self.data[idx_lastvib].split()[1])
        print(f"Found {self.nb_nm} frequencies in the Molden frequency file.")

        idx_freq = get_idx(self.data, "[FREQ]")[0] + 1
        freqs = get_col_array(self.data[idx_freq: idx_freq + self.nb_nm], 0)
        freqs = [float(item) for item in freqs]
        freqs = np.asarray(freqs)

        # Getting vibrations
        normal_modes = np.zeros((self.nb_nm, self.nb_cart))
        nm_indices = get_idx(self.data, "vibration")
        for nm, idx in enumerate(nm_indices):
            list3 = [get_col_array(self.data[idx+1: idx+1 + self.nb_atoms], n) for n in range(3)]
            grouped = [list3[i:i+3] for i in range(0, len(list3), 3)]
            for row_idx, (x, y, z) in enumerate(grouped):
                # Interleave the x, y, z values for each atom
                combined = np.column_stack([x, y, z]).flatten()

            normal_modes[nm] = combined

        return freqs, normal_modes

    def from_cart2nm(self, geom: np.array) -> np.array:
        """Converts the geometry in Cartesian coordinates [Bohr] to
        mass weigthed and frequency scaled normal modes as used in the vMCG formalism.

        Args:
            geom (np.array): Geometry in Cartesian coordinates. You need your data to be in [Bohr].
            You need to kabsch your data beforehand if necessary.

        Returns:
            normal modes coordinates.
        """
        return np.dot(self.cart2nm, (geom - self.geom_ref_bohr))

    def from_nm2cart(self, nm: np.array) -> np.array:
        """Converts from mass-frequency scaled normal modes to
        Cartesian coordinates [Bohr] as used in vMCG.
        
        Args:
            nm (normal modes coordinates): Geometry in normal mode coordinates.
        
        Returns:
            geometry in Cartesian coordinates.
        """
        return self.geom_ref_bohr + np.dot(self.nm2cart, nm)

    def check_RMSD(self, geom: np.array) -> float:
        """Calculates the Root Mean Square Deviation (RMSD) after converting the geometry to
        normal modes and back.
        
        Args:
            geom (np.array): Geometry in Cartesian coordinates.

        Returns:
            float: The RMSD value between the original geometry and the geometry after unit conversion.
        """

        geom_nm = self.from_cart2nm(geom)
        back_geom = self.from_nm2cart(geom_nm)

        return np.sqrt(((geom - back_geom) ** 2).mean())


class Frame:
    """Create a <Frame> object for each geometry at each timestep for a <Trajectory> object. """

    def __init__(self, geom: np.array, atoms: np.array, vel: np.array, forces: np.array, time: float,
                 ekin: float, epot: float, etot: float, eigenval: np.array, popul: np.array):
        """Initialize a Frame object.

            Args:
                geom (np.array): Geometry of the system (3nb_atoms). Units: Angstroms.
                atoms (np.array): Atomic symbols (nb_atoms). Units: str.
                vel (np.array): Velocities of atoms (nb_atoms, 3). Units: Bohr/a.u of time.
                forces (np.array): Forces on atoms (nb_atoms, 3). Units: TODO (add units in docstring when known).
                time (float): Current time. Units: femtosecond.
                ekin (float): Kinetic energy. Units: a.u. of energy.
                epot (float): Potential energy. Units: a.u. of energy.
                etot (float): Total energy. Units: a.u of energy.
                eigenval (np.array): Eigenvalues (nb_states,). Units: Bohr.
                popul (np.array): Electronic populations (nb_states,). Units: float.

        """ 

        # Arrays
        self.geometry = geom            # dim (3*nb_atoms)  [Angstroms]
        self.velocities = vel           # dim (nb_atoms, 3) [Bohr/a.u. of time]
        self.forces = forces            # dim (nb_atoms, 3) [???]  # TODO units of force
        self.eigenvalues = eigenval     # dim (nb_states,)  [Bohr]
        self.elec_populations = popul   # dim (nb_states,)  [float]
        self.atoms = atoms              # dim (nb_atoms,)   [str]

        # Values (float/int)
        self.time = time                # [femtosecond]
        self.energy_kin = ekin          # [a.u. of energy]
        self.energy_pot = epot          # [a.u. of energy]
        self.energy_tot = etot          # [a.u. of energy]
        self.state = list(eigenval).index(self.energy_pot)

    @property
    def e_kin_per_atom(self):
        """Calculate the Kinetic energy per atom.

        Units: Bohr

        Returns:
            np.array: Kinetic energy per atom.
        """
        # Get masses in electron mass and reshape as column vector
        masses = np.asarray([ELEMENTS_ATOMIC_MASS[s] for s in self.atoms]) * AMU_TO_ME
        masses = masses.reshape(-1, 1).T

        # Calculate norm of velocities and reshape as column vector
        v_norm = np.linalg.norm(self.velocities, axis=1).reshape(-1, 1).T

        # Calculates Atomic Kinetic Energy
        E_kin_atomic = 0.5 * masses * (v_norm ** 2)

        return E_kin_atomic.ravel()

    def measure_bond(self, *atoms: int, print_info: bool = False) -> float:
        """Measure a bond length between 2 atoms.

        Args:
            *atoms (int): Indices of the two atoms to measure the bond between (1-indexed).
            print_info (bool, optional): If True, prints bond information. Defaults to False.

        Returns:
            float: Bond length in Angstroms.
        """
        at1, at2 = atoms

        r1 = self.geometry[3 * (at1 - 1): 3 * (at1 - 1) + 3]
        r2 = self.geometry[3 * (at2 - 1): 3 * (at2 - 1) + 3]

        if print_info:
            print(f"{self.atoms[at1 - 1] + str(at1)}-{self.atoms[at2 - 1] + str(at2)} Bond Length (Angstrom)")

        return np.linalg.norm(r1 - r2)

    def measure_angle(self, *atoms: int, print_info: bool = False) -> float:
        """Measure an angle between 3 atoms.

        Args:
            *atoms (int): Indices of the three atoms to measure the angle between (1-indexed).
            print_info (bool, optional): If True, prints angle information. Defaults to False.

        Returns:
            float: Angle in degrees.
        """
        at1, at2, at3 = atoms

        r1 = self.geometry[3 * (at1 - 1):3 * (at1 - 1) + 3]
        r2 = self.geometry[3 * (at2 - 1):3 * (at2 - 1) + 3]
        r3 = self.geometry[3 * (at3 - 1):3 * (at3 - 1) + 3]

        if print_info:
            first = self.atoms[at1 - 1] + str(at1)
            second = self.atoms[at2 - 1] + str(at2)
            third = self.atoms[at3 - 1] + str(at3)
            print("{first}-{second}-{third} Angle (Degrees)")

        return angle(r1, r2, r3)

    def measure_dihedral(self, *atoms: int, print_info: bool = False) -> float:
        """Measure a dihedral angle between 4 atoms.

        Args:
            *atoms (int): Indices of the four atoms to measure the dihedral angle between (1-indexed).
            print_info (bool, optional): If True, prints dihedral angle information. Defaults to False.

        Returns:
            float: Dihedral angle in degrees.
        """
        at1, at2, at3, at4 = atoms

        r1 = self.geometry[3 * (at1 - 1): 3 * (at1 - 1) + 3]
        r2 = self.geometry[3 * (at2 - 1): 3 * (at2 - 1) + 3]
        r3 = self.geometry[3 * (at3 - 1): 3 * (at3 - 1) + 3]
        r4 = self.geometry[3 * (at4 - 1): 3 * (at4 - 1) + 3]

        if print_info:
            first = self.atoms[at1 - 1] + str(at1)
            second = self.atoms[at2 - 1] + str(at2)
            third = self.atoms[at3 - 1] + str(at3)
            fourth = self.atoms[at4 - 1] + str(at4)
            print(f"{first}-{second}-{third}-{fourth} Dihedral Angle (Degrees)")

        return dihedral(r1, r2, r3, r4)

    def create_geom(self, outname: str = None) -> str:
        """Print or create an xyz geometry file from a Frame.

        Args:
            outname (str, optional): Name of the output xyz file. If None, prints to stdout. Defaults to None.

        Returns:
            str: Filename of the created xyz file if outname is provided, otherwise None.
        """
        if outname:
            with open(f"{outname}.xyz", "w") as xyzfile:
                xyzfile.write(f"{len(self.atoms)}\n")
                xyzfile.write("\n")
                for a in range(len(self.atoms)):
                    atom_str = self.atoms[a]
                    atom_str += "    " + "      ".join([str(a) for a in self.geometry[3*a: 3*a+3]]) + "\n"
                    xyzfile.write(atom_str)
            return f"{outname}.xyz"

        atom_str = ""
        for a in range(len(self.atoms)):
            atom_str = self.atoms[a]
            atom_str += str(a + 1) + "    " + "      ".join([str(a) for a in self.geometry[3*a: 3*a+3]])
            print(atom_str)


class Trajectory:
    """Parent class for Molcas, Sharc and NX trajectories.

    This class serves as a base class for handling trajectory data
    from different programs (Molcas, Sharc, and NX).
    It provides common attributes and methods for accessing and processing
    trajectory information.
    """
    NAME = (None,)  # tuple[str]: Tuple of the program names associated with this trajectory type
    ETOT_THRESH = ETOT_THRESH  # float: Energy threshold for energy conservation checks [eV]

    def __init__(self):
        """Initialize a Trajectory object.

        Sets up the basic attributes for storing trajectory data,
        including:
        - trajectory number
        - number of steps
        - file keys
        - frames
        - number of states
        - number of atoms
        - energy conservation index

        It also initializes internal lists to store lines from output files.
        """
        self.traj_no = 0
        self.nb_steps = 0
        self.file_keys = []

        self.frames = []
        self.nb_ci = 0
        self.nb_atoms = 0
        self.en_conserved = True

        self._out_lines = []
        self._ener_lines = []
        self._xyz_lines = []

    @classmethod
    def get_parser(cls, prog: str) -> type(Trajectory):
        """Get the proper child parser class for a given program name.

        This class method iterates through the subclasses of `Trajectory`
        and returns the subclass whose `NAME` attribute contains the
        provided program name. This allows for dynamic selection of the
        appropriate parser based on the program that generated the trajectory file.

        Args:
            prog (str): The name of the program ('MOLCAS', 'SHARC', 'NX').

        Returns:
            type[:py:class:`Trajectory`]: The subclass of `Trajectory` corresponding to the program name.

        Raises:
            Exception: If the program name is not recognized among the supported programs.
        """
        for subclass in cls.__subclasses__():
            if prog in subclass.NAME:
                return subclass
            
        raise Exception(f"Program not recognised (TSH programs = {' ,'.join(VALID_PROGRAMS)}).")

    def _read_output(self, *args):
        raise NotImplementedError

    def _get_elec_pop(self) -> np.array:
        raise NotImplementedError

    def _get_velocities(self) -> np.array:
        """Get velocities in inherited classes. """
        raise NotImplementedError

    def _get_forces(self, *args) -> np.array:
        """Get forces in inherited classes. """
        raise NotImplementedError

    def _get_eigenvalues(self) -> np.array:
        raise NotImplementedError

    def _get_information(self, eigenvalues: np.array) -> Tuple:
        raise NotImplementedError

    def _get_geom(self) -> List:
        """Fetch geometries from xyz file [Angstroms] as list of numpy array.
        
        Returns:
            The list of geometries visited along the trajectory.
        """

        geom_all = []
        for step in range(self.nb_steps):
            start = step * (2 + self.nb_atoms) + 2
            end = step * (2 + self.nb_atoms) + 2 + self.nb_atoms
            geom_string = self._xyz_lines[start: end]

            geom = []
            for i in range(self.nb_atoms):
                geom.append(geom_string[i][3:].split())

            reshape = np.array([np.array([x for x in geom]).astype(float)]).reshape(self.nb_atoms*3)
            geom_all.append(reshape)

        return geom_all

    def _set_frames(self):
        """This method calculates and sets the `frames` attribute of the class instance.
        It populates the `frames` list with `Frame` objects, each representing a snapshot
        of the trajectory at a given time step.

        The method first retrieves necessary data such as geometry, electronic populations,
        velocities, forces, and eigenvalues from other methods of the class.
        It then checks for energy conservation throughout the trajectory. If the total
        energy deviates from the initial total energy by more than a predefined
        threshold (`self.ETOT_THRESH`), the energy is considered not conserved.
        In this case, the `en_conserved` flag is set to `False`, and the method
        returns without creating or populating the `frames` list.

        If energy is conserved within the threshold, the method proceeds to create
        `Frame` objects for each time step. Each `Frame` object encapsulates the
        geometry, atom types, velocities, forces, time, kinetic energy, potential energy,
        total energy, eigenvalues, and electronic populations at that specific time step.
        These `Frame` objects are then appended to the `self.frames` list.

        Returns:
            None. The method modifies the `self.frames` attribute in place.
            Returns early (without populating `self.frames`) if energy is not conserved.
        """
        # Geometry in Angstrom
        geom = self._get_geom()

        # Electronic populations (Aii=Ci*.Ci)
        elec_populations = self._get_elec_pop()

        # Velocities [Bohr/a.u.]
        velocities = self._get_velocities()

        # Forces
        forces = self._get_forces(elec_populations)

        # Eigenvalues
        eigenvalues = self._get_eigenvalues()

        # Atoms list
        atoms = np.asarray([line.split()[0] for line in self._xyz_lines[2: 2+self.nb_atoms]])

        # Get additional informations
        time, epot, ekin, etot = self._get_information(eigenvalues)

        # Check energy conservation (not first step)
        for i in range(self.nb_steps):
            if (abs(etot[i] - etot[0]) * HARTREE_TO_EV) > self.ETOT_THRESH and i >= 1:
                self.en_conserved = False
                return

        # Create frames
        for i in range(self.nb_steps):
            frame = Frame(
                geom=geom[i],
                atoms=atoms,
                vel=velocities[i],
                forces=forces[i],
                time=time[i],
                ekin=ekin[i],
                epot=epot[i],
                etot=etot[i],
                eigenval=eigenvalues[i],
                popul=elec_populations[i]
            )
            self.frames.append(frame)

    def _hop_information(self, state1: int, state2: int) -> List[int]:
        """ Gets information on when a hops occur (all possible hops) in trajectory.

        Args:
            TODO

        Returns:
            List[int]: A list of indices representing the time steps
                before a hop from `state1` to `state2` occurs in the trajectory.
                Returns an empty list if no such hop is found.
        """
        hopping_dict = {}
        for i in list(itertools.permutations(range(self.nb_ci), 2)):
            hopping_dict[i] = []

        for i in range(self.nb_steps-1):
            actual = self.frames[i].state
            next_ = self.frames[i+1].state
            if actual != next_:
                hopping_dict[(actual, next_)].append(i)

        return hopping_dict[(state1, state2)]

    def get_frame(self, idx: int) -> Frame:
        """Return the frame from index. """
        return self.frames[idx]

    def get_population_array(self) -> np.array:
        """Returns the electronic populations of all states throughout a trajectory. """
        populations = np.array([self.get_frame(0).elec_populations])
        for i in range(1, self.nb_steps):
            populations = np.append(populations, [self.get_frame(i).elec_populations], axis=0)

        return populations

    def get_active_state_array(self) -> np.array:
        """Returns array (analagous to <get_population_array>) with active state for each frame in <Trajectory>. """
        # Get population at first frame
        populations = np.array([self.get_frame(0).elec_populations])

        for i in range(1, self.nb_steps):
            array_formatted = [0] * self.nb_ci
            array_formatted[self.frames[i].state] = 1
            populations = np.vstack([populations, array_formatted])

        return populations

    def hop_frame_indices(self, state1: int, state2: int) -> List:
        """Returns hop indices between a given pair of states in given direction. """
        return self._hop_information(state1, state2)

    def hop_frames_actual(self, state1: int, state2: int) -> List:
        """Returns frames of hops for a given pair of states in given direction. """
        hop_geom_list = []
        for i in self._hop_information(state1, state2):
            hop_geom_list.append(self.frames[i])

        return hop_geom_list

    def create_trajectory(self, outname: str = "traj") -> str:
        """Create a xyz file from trajectory. """
        with open(f"{outname}_{self.traj_no}.xyz", "w") as xyzfile:
            for frame in self.frames:
                xyzfile.write(str(self.nb_atoms))
                xyzfile.write(f"\n time = {frame.time:.2f} fs \n")
                for a in range(len(frame.atoms)):
                    atom_str = frame.atoms[a]
                    atom_str += "    " + "      ".join([str(a) for a in frame.geometry[3 * a:3 * a + 3]]) + "\n"
                    xyzfile.write(atom_str)

        return f"{outname}.xyz"


class MolcasTraj(Trajectory):
    """Represents a Molcas trajectory.

    This class reads Molcas output files (energies and xyz coordinates)
    and creates a Trajectory object. It extracts information like number of steps,
    number of states, and number of atoms from the files.
    """
    NAME = ("molcas",)

    def __init__(self, traj_no: int, file_keys: List = None, max_time: int = None):
        """Initializes a MolcasTraj object.

        Reads Molcas output files and sets up the trajectory.

        Args:
            traj_no (int): The trajectory number (1, 2, 3, ...). It is used to construct the filenames.
            file_keys (List[str], optional): List of file keys. These are prefixes
                used to locate the Molcas output files. Necessary for Molcas only.
                Defaults to None.
            max_time (int, optional): Maximum number of time steps to read. If None,
                all time steps are read. Defaults to None.
        """
        super().__init__()

        self.traj_no = traj_no
        self.file_keys = file_keys

        # Read output
        self._read_output(max_time)

        # Set Frame
        self._set_frames()

        # Delete init variables to save memory and space
        del self._xyz_lines
        del self._out_lines
        del self._ener_lines

    def _read_output(self, max_time):
        """Reads and parses Molcas output files to fill trajectory attributes.

        This method reads the:
        - `.output`
        - `.md.energies`
        - `.md.xyz`
        files and extracts the number of steps, number of CI states, and number of atoms.
        It populates the attributes `_out_lines`, `_ener_lines`, `_xyz_lines`,
        `nb_steps`, `nb_ci`, and `nb_atoms`.

        Args:
            max_time (int): Maximum number of time steps to read. This parameter is
                actually used in the `__init__` method to limit the number of steps.
        """
        for key in self.file_keys:
            with open(f"{key}_{self.traj_no}.output") as fp:
                self._out_lines.extend(fp.read().splitlines())

        with open(f"{self.file_keys[-1]}_{self.traj_no}.md.energies") as fp:
            self._ener_lines = fp.read().splitlines()

        with open(f"{self.file_keys[-1]}_{self.traj_no}.md.xyz") as fp:
            self._xyz_lines = fp.read().splitlines()

        # Setup <nb_steps> (-1 for header)
        self.nb_steps = len(self._ener_lines)-1
        self.nb_steps = len(self._ener_lines)-1 if max_time is None else np.min((len(self._ener_lines)-1, max_time))

        # Setup <nb_ci> Number of eigenstates in CASSCF
        self.nb_ci = int(len(self._ener_lines[1].split())-4)

        # Number of atoms in molecule
        self.nb_atoms = int(self._xyz_lines[0])

    def _get_information(self, eigenvalues: np.array) -> Tuple:
        """Get time, potential energy, kinetic energy, and total energy.
        Time is in [fs] and energies are in [Bohr]. """
        # Time [fs]
        time = AU_TO_FS * get_col_array(self._ener_lines[1:], 0)
        epot = get_col_array(self._ener_lines[1:], 1)
        ekin = get_col_array(self._ener_lines[1:], 2)
        etot = get_col_array(self._ener_lines[1:], 3)

        return time, epot, ekin, etot

    def _get_elec_pop(self) -> np.array:
        """Get the electronic population of each state from output file. """
        all_pop = []
        pop_indices = get_idx(self._out_lines, "Gnuplot")
        for idx in pop_indices:
            pop_string = " ".join(self._out_lines[idx: idx + int((self.nb_ci/3)+2)])
            elec_pop = np.asarray(pop_string.split()[1: self.nb_ci+1]).astype(float)
            all_pop.append(elec_pop)

        return np.array(all_pop)

    def _get_eigenvalues(self) -> np.array:
        """Get the eigenvalues raw (nb_ci, nb_steps) from output file to transposed (nb_steps, nb_ci).
        Eigenvalues in [Bohr]. """
        eigenvals_raw = []
        for ci in range(self.nb_ci):
            eigenvals_raw.append(get_col_array(self._ener_lines[1:], ci + 4))

        return np.array(eigenvals_raw).T

    def _get_velocities(self) -> np.array:
        """Get the velocities [Bohr/a.u.] values for each atom from output file.
        Check if the reduced velocity exists, and, if so take it instead. """
        red = get_idx(self._out_lines, "Vel (red dim)")
        indices = get_idx(self._out_lines, "Velocities      (time")
        if len(red) != 0:
            indices.pop(0)
            indices = red + indices
        
        vel_all = []
        for idx in indices:
            vel_list = self._out_lines[idx + 4: idx + 6 + self.nb_atoms]
            
            for i in range(len(vel_list)-1,-1,-1):
                elem = vel_list[i]
                if len(elem.split()) == 0 or \
                    "Note" in elem or \
                    "----------------" in elem or \
                    "security" in elem or \
                    "warnings" in elem:
                    vel_list.pop(i)

            vel_x = get_col_array(vel_list, 2)
            vel_y = get_col_array(vel_list, 3)
            vel_z = get_col_array(vel_list, 4)
            vel = np.array([vel_x, vel_y, vel_z]).T
            vel_all.append(vel)

        return np.array(vel_all)

    def _get_forces(self, *args) -> np.array:
        """Get the forces values for each atom from output file. """
        # Get indices
        indices = get_idx(self._out_lines, "Old Coordinates (time")

        force_all = []
        for idx in indices:
            force_list = self._out_lines[idx + 4: idx + 5 + self.nb_atoms]

            for i, elem in enumerate(force_list):
                if "Note" in elem or "----------------" in elem:
                    force_list.pop(i)
                    if "Note" in elem:
                        print(f"THIS WAS POPPED FROM FORCES: {elem}")

            force_x = get_col_array(force_list, 6)
            force_y = get_col_array(force_list, 7)
            force_z = get_col_array(force_list, 8)
            force = np.array([force_x, force_y, force_z]).T
            force_all.append(force)

        return np.array(force_all)


class SharcTraj(Trajectory):
    """Represents a SHARC trajectory.

    This class reads Sharc output file and creates a Trajectory object.
    It extracts information like number of steps, number of states, and number of atoms from the files.
    """
    NAME = ("sharc",)

    def __init__(self, traj_no: int, file_keys: List = None, max_time: int = None):
        """Initializes a SharcTraj object.

        Reads SHARC output files and sets up the trajectory.

        Args:
            traj_no (int): The trajectory number (1, 2, 3, ...). It is used to construct the filenames.
            file_keys (List[str], optional): List of file keys. These are prefixes
                used to locate the Molcas output files. Necessary for Molcas only.
                Defaults to None.
            max_time (int, optional): Maximum number of time steps to read. If None,
                all time steps are read. Defaults to None.
        """
        super().__init__()

        # set Attributes
        self.traj_no = traj_no

        # Read output
        self._read_output(max_time)

        # Set <Frames> object to <Trajectory>
        self._set_frames()

        # Delete init variables to save memory and space
        del self._xyz_lines
        del self._ener_lines
        del self._out_lines

    def _read_output(self, max_time):
        """Reads and parses SHARC output files to fill trajectory attributes.

        This method reads the:
        - `output.dat`
        - `output.lis`
        - `output.xyz`
        files and extracts the number of steps, number of CI states, and number of atoms.
        It populates the attributes `_out_lines`, `_ener_lines`, `_xyz_lines`,
        `nb_steps`, `nb_ci`, and `nb_atoms`.

        Args:
            max_time (int): Maximum number of time steps to read. This parameter is
                actually used in the `__init__` method to limit the number of steps.
        """
        # with open(f"TRAJ{self.traj_no}/output.dat") as datfile:  # Previous directory nomenclature
        with open("TRAJ_%s/output.dat" % str(self.traj_no).zfill(4)) as datfile:  # Directory nomenclature
            self._out_lines = datfile.read().splitlines()

        # with open(f"TRAJ{self.traj_no}/output.lis") as lisfile:  # Previous directory nomenclature
        with open("TRAJ_%s/output.lis" % str(self.traj_no).zfill(4)) as lisfile:  # Directory nomenclature
            for line in lisfile.read().splitlines():
                if line[0] != "#":
                    self._ener_lines.append(line)

        # with open(f"TRAJ{self.traj_no}/output.xyz") as xyzfile:  # Previous directory nomenclature
        with open("TRAJ_%s/output.xyz" % str(self.traj_no).zfill(4)) as xyzfile:  # Directory nomenclature
            self._xyz_lines = xyzfile.read().splitlines()

        # Add <nb_steps> for this trajectory
        self.nb_steps = len(self._ener_lines) if max_time is None else np.min((len(self._ener_lines), max_time))

        # Add number of eigenstates in CASSCF
        for line in self._out_lines[:20]:
            if line.split()[0] == "nstates_m":
                state_list = line.split()[1:]
                state_list = [int(i) for i in state_list]

                state_count = 0
                for multiplicity, count in enumerate(state_list):
                    state_count += (multiplicity + 1) * count
                self.nb_ci = state_count

        # Number of atoms in molecule
        self.nb_atoms = int(self._xyz_lines[0])

    def _get_information(self, eigenvalues: np.array) -> Tuple:
        """Get time, potential energy, kinetic energy, and total energy.
        
        Time is in [fs] and energies are in [Bohr].
        
        Args:
            eigenvalues (np.array): TODO

        Returns:
            time (np.array): Time 
            epot (np.array): Potential energy 
            ekin (np.array): Kinetic energy 
            etot (np.array): Total (Kinetic + Potential) energy
        """
        # Time in fs
        time = get_col_array(self._ener_lines, 1)

        ekin, state_list = [], []
        for i, line in enumerate(self._out_lines):
            if "! 7 Ekin" in line:
                ekin.append(float(self._out_lines[i+1]))
            elif "! 8 states" in line:
                state_list.append(int(self._out_lines[i+1].split()[1]))

        ekin = np.array(ekin)
        epot = np.array([eigenvalues[i][state_list[i]-1] for i in range(len(eigenvalues))])
        etot = epot + ekin

        return time, epot, ekin, etot

    def _get_elec_pop(self) -> np.array:
        """Get the electronic population of each state from output file. """
        coeffs_arr = []
        for i, line in enumerate(self._out_lines):
            if "! 5 Coefficients" in line:
                coeffs = []
                for ci in range(self.nb_ci):
                    pop_0, pop_1 = map(float, self._out_lines[i + ci + 1].split())
                    coeffs.append(pop_0 ** 2 + pop_1 ** 2)
                coeffs_arr.append(coeffs)

        return np.array(coeffs_arr)

    def _get_eigenvalues(self) -> np.array:
        """Get the eigenvalues [Bohr] along a trajectory. """
        eigenvalues = []
        for i, line in enumerate(self._out_lines):
            if "! 1 Hamiltonian (MCH)" in line:
                ham_lines = self._out_lines[i + 1: i + self.nb_ci + 1]
                eigenvalues.append([float(ham_lines[state].split()[state*2]) for state in range(self.nb_ci)])
        eigenvalues = np.array(eigenvalues)

        return eigenvalues

    def _get_velocities(self) -> np.array:
        """Get the velocities along a trajectory in [Bohr/a.u.]. """
        mom_all = []
        for i, line in enumerate(self._out_lines):
            if "! 12 Velocities" in line:
                vel_lines = self._out_lines[i + 1: i + self.nb_atoms + 1]
                vel_lines = np.asarray([atom.split() for atom in vel_lines], dtype=float).reshape(self.nb_atoms*3)
                mom_all.append(vel_lines)

        return mom_all

    def _get_forces(self, states) -> np.array:
        """Get gradient of the active state in diagonal representation  # TODO Units of force
        SHARC's `no_grad_correct` scheme is used when calculating diagonal gradients.
        
        Args:
            states: TODO
        
        Returns:
            forces (np.array): TODO
        """
        force_mch_all = []
        u_matrix_all = []
        force_all = []

        for i, line in enumerate(self._out_lines):
            if "! 2 U matrix" in line:
                ham_lines = self._out_lines[i + 1: i + self.nb_ci + 1]
                u_matrix_real = []
                u_matrix_imag = []
                for state in ham_lines:
                    u_matrix_real.append(state.split()[0::2])
                    u_matrix_imag.append(state.split()[1::2])
                u_matrix = np.empty((self.nb_ci, self.nb_ci), dtype=complex)
                u_matrix.real = u_matrix_real
                u_matrix.imag = u_matrix_imag
                u_matrix_all.append(u_matrix)

            if "! 15 Gradients (MCH)" in line:
                grad_lines = self._out_lines[i + 1: i + self.nb_atoms + 1]
                grad_lines = [atom.split() for atom in grad_lines]
                grad_lines = np.asarray(grad_lines, dtype=float)
                force_mch_all.append(grad_lines)

        for i, matrix in enumerate(u_matrix_all):
            gradient_matrix = np.zeros((self.nb_ci, self.nb_ci, self.nb_atoms, 3))
            if force_mch_all != []:
                for j in np.arange(self.nb_ci):
                    gradient_matrix[j, j] = force_mch_all[i * self.nb_ci + j]
            if force_mch_all == []:
                for j in np.arange(self.nb_ci):
                    gradient_matrix[j, j] = 0.0

            gradient_diag = np.tensordot(matrix.T.conjugate(), gradient_matrix, axes=1)
            gradient_diag = np.tensordot(gradient_diag, matrix, axes=(1, 0))
            gradient_diag = np.moveaxis(gradient_diag, -1, 0)

            state, = np.where(states[i] == np.max(states[i]))[0]

            # Active state selection
            force = gradient_diag[state, state]
            force = force.real.reshape(self.nb_atoms * 3)

            force_all.append(force)

        return np.array(force_all)


class NXTraj(Trajectory):
    """Represents a NX trajectory.

    This class reads NX output file and creates a Trajectory object.
    It extracts information like number of steps, number of states, and number of atoms from the files.
    """

    NAME = ("nx",)

    def __init__(self, traj_no: int, file_keys: List = None):
        """Initializes a NXTraj object.

        Reads NX output files and sets up the trajectory.

        Args:
            traj_no (int): The trajectory number (1, 2, 3, ...). It is used to construct the filenames.
            file_keys (List[str], optional): List of file keys. These are prefixes
                used to locate the Molcas output files. Necessary for Molcas only.
                Defaults to None.
        """
        super().__init__()

        # set Attributes
        self.traj_no = traj_no

        # Read output
        self._read_output()

        # Set Frame
        self._set_frames()

        # Delete init variables to save memory and space
        del self._xyz_lines
        del self._ener_lines
        del self._out_lines

    def _read_output(self):
        """Reads and parses NX output files to fill trajectory attributes.

        This method reads the:
        - `en.dat`
        - `dyn.out`
        - `dyn.xyz`
        files and extracts the number of steps, number of CI states, and number of atoms.
        It populates the attributes `_out_lines`, `_ener_lines`, `_xyz_lines`,
        `nb_steps`, `nb_ci`, and `nb_atoms`.

        Args:
            max_time (int): Maximum number of time steps to read. This parameter is
                actually used in the `__init__` method to limit the number of steps.
        """

        with open(f"TRAJ{self.traj_no}/RESULTS/en.dat") as fp:
            self._ener_lines = fp.read().splitlines()

        with open(f"TRAJ{self.traj_no}/RESULTS/dyn.out") as fp:
            self._out_lines = fp.read().splitlines()

        self._create_xyz_geom()
        with open(f"TRAJ{self.traj_no}/RESULTS/dyn.xyz") as fp:
            self._xyz_lines = fp.read().splitlines()

        # Populate <nb_steps>
        self.nb_steps = len(self._ener_lines)

        # Number of atoms in molecule
        self.nb_atoms = int(self._xyz_lines[0])

        # Number of eigenstates in CASSCF
        with open(f"TRAJ{self.traj_no}/RESULTS/nx.log") as fp:
            for line in fp:
                if "nstat     =" in line:
                    self.nb_ci = int(line.split()[-1])
                    break

    def _create_xyz_geom(self):
        """Create a <dyn.xyz> file [Angstroms] for each <dyn.out> file encountered.

        This method reads the 'dyn.out' file from the specified trajectory directory
        and extracts geometry information to create a corresponding 'dyn.xyz' file.
        The coordinates in the output '.xyz' file are converted from Bohr to Angstroms.

        Args:
            self: The instance of the class. The `traj_no` attribute is used to
                construct the input and output file paths.

        Returns:
            None. The function writes the geometry data to a file.
        """
        input_file = f"TRAJ{self.traj_no}/RESULTS/dyn.out"
        output_file = f"TRAJ{self.traj_no}/RESULTS/dyn.xyz"

        def fetch_geometries(input_file: str) -> Tuple[float, List]:
            """Get each geometry from <dyn.out> file.

            This function parses a '.dyn.out' file to extract geometry data at each time step.
            It reads the file line by line, identifies geometry sections based on keywords
            "TIME =" and "geometry", and extracts atom coordinates. The coordinates
            are converted from Bohr to Angstroms using the `BOHR_TO_ANG` constant.

            Args:
            input_file (str): The path to the '.dyn.out' input file.

            Returns:
            Tuple[float, List[List[List[float]]]]: A tuple containing:
                - float: The last time value read from the file.
                - List[List[List[float]]]: A list of geometries. Each geometry is a list
                of atoms, and each atom is represented as a list:
                `[time, element, x, y, z]`, where x, y, and z are coordinates in Angstroms.
            """
            with open(input_file) as fp:
                data = fp.readlines()

            geometry_section = False
            geometries = []
            current_geometry = []
            time = 0
            for line in data:
                if "TIME =" in line:
                    time = float(line.split()[-2])
                elif "geometry" in line:
                    geometry_section = True

                if geometry_section and re.match(r"^ [A-Za-z]", line):
                    atom_data = line.strip().split()
                    element = atom_data[0]
                    x = float(atom_data[2]) * BOHR_TO_ANG
                    y = float(atom_data[3]) * BOHR_TO_ANG
                    z = float(atom_data[4]) * BOHR_TO_ANG
                    current_geometry.append([time, element, x, y, z])

                if "velocity" in line:
                    geometry_section = False
                    geometries.append(current_geometry)
                    current_geometry = []

            return time, geometries

        def write_to_file(output_file: str, time: float, geometries: List):
            """Writes geometry data to file.

            Args:
                output_file (str): The path to the file to write to.
                time (float): A time value (TODO or clean).
                geometries (List): A list of geometries.
            """
            with open(output_file, "w") as fp:
                for geom in geometries:
                    fp.write(str(len(geom)) + "\n")
                    fp.write(f"TIME = {geom[0][0]:.2f}\n")
                    for atom in geom:
                        fp.write(f"{atom[1]}\t\t{atom[2]:.8f}\t\t{atom[3]:.8f}\t\t{atom[4]:.8f}\n")

        write_to_file(output_file, *fetch_geometries(input_file))

    def _get_information(self, eigenvalues: np.array) -> Tuple:
        """Get time, potential energy, kinetic energy, and total energy.

        Time is in [fs] and energies are in [Bohr].

        Args:
            eigenvalues (np.array): TODO or clean

        Returns:
            time (np.array): Time
            epot (np.array): Potential energy
            ekin (np.array): Kinetic energy
            etot (np.array): Total (Kinetic + Potential) energy
        """
        # Time in fs
        time = get_col_array(self._ener_lines, 0)
        epot = get_col_array(self._ener_lines, self.nb_ci+1)
        etot = get_col_array(self._ener_lines, self.nb_ci+2)
        ekin = etot - epot

        return time, epot, ekin, etot

    def _get_elec_pop(self) -> np.array:
        """Get the electronic population of each state from output file. """
        coeffs_arr = []
        for i, line in enumerate(self._out_lines):
            if "Wave function state  1:" in line:
                coeffs = []
                for ci in range(self.nb_ci):
                    pop_0 = float(self._out_lines[i + ci].split()[-2])
                    pop_1 = float(self._out_lines[i + ci].split()[-1])
                    coeffs.append(pop_0 ** 2 + pop_1 ** 2)
                coeffs_arr.append(coeffs)

        return np.array(coeffs_arr)

    def _get_eigenvalues(self) -> np.array:
        """Get the eigenvalues [Bohr] along a trajectory. """
        eigenvals_raw = []
        for ci in range(self.nb_ci):
            eigenvals_raw.append(get_col_array(self._ener_lines, ci+1))

        return np.asarray(eigenvals_raw).T

    def _get_velocities(self) -> np.array:
        """Get the velocities along a trajectory in [Bohr/a.u.]. """
        mom_all = []
        indices = get_idx(self._out_lines, "velocity:")
        for idx in indices:
            mom_list = self._out_lines[idx + 1: idx + 1 + self.nb_atoms]
            mom_x = get_col_array(mom_list, 0)
            mom_y = get_col_array(mom_list, 1)
            mom_z = get_col_array(mom_list, 2)
            mom = np.array([mom_x, mom_y, mom_z]).T
            mom_all.append(mom)

        return np.asarray(mom_all)


class Ensemble:
    """Create an Ensemble object of trajectories. """

    def __init__(self, prog: str, trajs_idx: List, file_keys: List = None, max_time: int = None,
                 en_thresh: float = ETOT_THRESH):
        """
        Initialize an Ensemble object.

        Sets up the basic attributes for storing trajectory data,
        including:
        - prog (str): the parser to chose
        - trajs_idx (List): list of the trajectories to include for this ensemble
        - file_keys (List): list of the root names (necessary for Molcas only traj_x_1, traj_x_2, ...)
        - max_time (int): Last time to read in the trajectories. Maximum time defaults to None
        - en_thresh (float): energy conservation threshold in eV
        """
        # Check program in the VALID_PROGRAMS List
        if prog.lower() not in VALID_PROGRAMS:
            raise Exception(f"Program not recognised (TSH programs = {', '.join(VALID_PROGRAMS)}")

        # Add prog to instance attributes
        self.prog = prog.lower()
        self.en_thresh = en_thresh

        # Add traj indices not added to <Ensemble>
        self.trajs_no_not_added = []

        # Add maximum time
        self.max_time = max_time

        # Create trajs list attribute containing <Trajectory> object
        self._create_trajs(trajs_idx, file_keys)

        # Check <en_conserved> True for each <Trajectory>
        self._check_energy_conservation()

        # Check <nb_steps> same for all trejectories
        self._check_nb_steps()

        # Get information from first <Trajectory> in list
        self.timestep = self.trajs[0].frames[1].time
        self.nb_steps = self.trajs[0].nb_steps
        self.total_time = self.nb_steps * self.timestep
        self.nb_ci = self.trajs[0].nb_ci

        # Print <Ensemble> info
        self.print_ensemble_info()

    @property
    def nb_trajs(self) -> int:
        """Returns the number of trajectories. """
        return len(self.trajs)

    @property
    def av_adiabatic_pop(self) -> np.array:
        """Change a function as an attribute - calculated everytime called. """
        adiabatic_pop = np.zeros((self.nb_steps, self.nb_ci))
        for traj in self.trajs:
            adiabatic_pop += traj.get_population_array()
        adiabatic_pop /= self.nb_trajs

        return adiabatic_pop

    @property
    def av_active_state(self) -> np.array:
        """Change a function as an attribute - calculated everytime called. """
        av_active_state = np.zeros((self.nb_steps, self.nb_ci))
        for traj in self.trajs:
            av_active_state += traj.get_active_state_array()
        av_active_state /= self.nb_trajs

        return av_active_state

    def _create_trajs(self, trajs_idx: List, file_keys: List):
        """Create trajectory list attribute containing <Trajectory> object.
        
        Args:
            - trajs_idx (List): a list of trajectories indeces to include in the Ensemble
            - file_keys (List): list of the root names (necessary for Molcas only traj_x_1, traj_x_2, ...)

        Returns:
            None, it populates self.trajs
        """
        # Get parser (Molcas, SHARC, NX)
        Traj = Trajectory.get_parser(prog=self.prog.lower())
        Traj.ETOT_THRESH = self.en_thresh

        self.trajs = []
        for idx in tqdm(trajs_idx, ncols=70):
            traj = Traj(traj_no=idx, file_keys=[f"TRAJ{idx}/{key}" for key in file_keys] if file_keys else None,
                        max_time=self.max_time)
            self.trajs.append(traj)

    def _check_energy_conservation(self):
        """Check that <en_conserved> is True for each <Trajectory> in self.trajs else remove it. """
        traj_to_remove = []
        for traj in self.trajs:
            if not traj.en_conserved:
                self.trajs_no_not_added.append(traj.traj_no)
                traj_to_remove.append(traj)

        for traj in traj_to_remove:
            self.trajs.remove(traj)
            print(f"TRAJ ({traj.traj_no}) was removed because of energy conservation (> {self.en_thresh} eV).")

        # Check still traj in self.trajs
        if len(self.trajs) == 0:
            print("All trajectories were removed because of energy conservation. Ensemble not created.")
            print(f"En. conservation threshold: {self.en_thresh} eV")
            exit()

    def _check_nb_steps(self):
        """Removes trajectories that crashed before the end. """
        # Find maximum <nb_steps>
        max_nb_steps = max(traj.nb_steps for traj in self.trajs)

        # Put <Trajectory> indices if <nb_steps> != <max_nb_steps>
        for traj in self.trajs:
            if traj.nb_steps != max_nb_steps:
                self.trajs_no_not_added.append(traj.traj_no)
                print(f"TRAJ ({traj.traj_no}) was removed because its nb steps ({traj.nb_steps}) < "
                      f"max_nb_steps ({max_nb_steps})")

        # Update trajs attribute
        self.trajs = [traj for traj in self.trajs if traj.nb_steps == max_nb_steps]

    def print_ensemble_info(self):
        """Print <Ensemble> information. """
        headers = ["Info", "Value(s)"]

        data = [
            ["Prog. used", f"{self.prog.capitalize()}"],
            ["En. threshold", f"{self.en_thresh} eV"],
            ["Nb trajs", f"{self.nb_trajs}"],
            ["Nb trajs removed", f"{len(self.trajs_no_not_added)}"],
            ["Trajs removed", f"{self.trajs_no_not_added}"],
            ["Timestep", f"{self.timestep} fs"],
            ["Nb steps", f"{self.nb_steps-1}"],
            ["Total sim. time", f"{self.total_time-self.timestep} fs"],
        ]

        # Calculate each column max width
        col_widths = [max(len(item) for item in col) for col in zip(*data, headers)]

        # Print headers
        header_row = " | ".join(format(header, f"{width}s") for (header, width) in zip(headers, col_widths)) + " |"
        print("")
        print(header_row)
        print("-" * len(header_row))

        # Print data rows
        for row in data:
            row_str = " | ".join(format(item, f"{width}s") for (item, width) in zip(row, col_widths)) + " |"
            print(row_str)
        print("")

    def get_traj(self, traj_no: int) -> Trajectory:
        """Get a trajectory based on the <traj_no>. Begin at one (Traj1, Traj2, Traj3, ...).
        
        Args:
            - traj_no (int): The index that identifies a trajectory

        Returns:
            traj (Trajectory) object

        """
        try:
            traj = [traj for traj in self.trajs if traj.traj_no == traj_no][0]
        except IndexError:
            raise IndexError(f"This traj ({traj_no}) does not exist in this ensemble.")

        return traj

    def get_trajs_state_list(self, state: int) -> List:
        """Return a list of all trajectories number with a specified state. """
        return [traj.traj_no for traj in self.trajs if traj.get_frame(-1).state == state]

    def calc_states_dict(self) -> Dict[str, List]:
        """Return a dict containing as values a list of all states for each frame for each trajectory. """
        state_dict = {}
        for traj in self.trajs:
            state_list = []
            for frame in traj.frames:
                state_list.append(frame.state)
            state_dict[f"traj_{traj.traj_no}"] = state_list

        return state_dict

    def calc_geom_properties(self, *atoms: int, **kwargs) -> Dict[str, np.array]:
        """Calculates geometric properties (distances, valence or dihedral angles).

        Args:
            - atoms (int): Indices of the atoms to measure the geometric parameter (1-indexed).
        
        Returns:
            Dict containing as keys the trajectory indeces ("traj_{traj.traj_no}") and
            as values a np.array of the geometric property requested for each trajectory.
        """
        # Check if *atoms is properly given
        if len(atoms) not in (2, 3, 4):
            raise ValueError("Invalid number of atoms. Supported geometries are bond lengths (2 atoms), "
                             "angles (3 atoms), and dihedrals (4 atoms).")

        geometries = {}
        for traj in self.trajs:
            traj_geometries = []
            for frame in traj.frames:
                if len(atoms) == 2:
                    traj_geometries.append(frame.measure_bond(*atoms, **kwargs))
                elif len(atoms) == 3:
                    traj_geometries.append(frame.measure_angle(*atoms, **kwargs))
                elif len(atoms) == 4:
                    traj_geometries.append(frame.measure_dihedral(*atoms, **kwargs))
            geometries[f"traj_{traj.traj_no}"] = np.asarray(traj_geometries)

        return geometries

    def get_all_geometries(self, as_list: bool = False, flat: bool = False) -> Union[np.array, Dict[str, List]]:
        """ Collects all the geometries [Angstroms] in each <Frame> of each <Trajectory>.
        
        Args:
            - as_list (bool): If set as True prepares a np.array of all the geometries. Defaults to False.
            - flat (bool): Defaults to False.

        Returns:
            geom_dict (Dict): Dict containing all the geometries (as flat array if flat is set as True).
        """
        if as_list:
            geom_list = []
            for traj in self.trajs:
                for frame in traj.frames:
                    geom_list.append(frame.geometry.ravel() if flat else frame.geometry)
            return np.asarray(geom_list)

        geom_dict = {}
        for traj in self.trajs:
            geom_dict[f"traj_{traj.traj_no}"] = []
            for frame in traj.frames:
                geom_dict[f"traj_{traj.traj_no}"].append(frame.geometry.ravel() if flat else frame.geometry)

        return geom_dict

    def get_all_forces(self, as_list: bool = False) -> Union[np.array, Dict[str, List]]:
        """ Collects all the forces [TODO] in each <Frame> of each <Trajectory>.
        
        Args:
            - as_list (bool): If set as True prepares a np.array of all the forces. Defaults to False.

        Returns:
            geom_dict (Dict): Dict containing all the forces as a flat (3N,) array.
        """
        if as_list:
            forces_list = []
            for traj in self.trajs:
                for frame in traj.frames:
                    forces_list.append(frame.forces.ravel())
            return np.asarray(forces_list)

        forces_dict = {}
        for traj in self.trajs:
            forces_dict[f"traj_{traj.traj_no}"] = []
            for frame in traj.frames:
                forces_dict[f"traj_{traj.traj_no}"].append(frame.forces.ravel())

        return forces_dict

    def return_hop_frames(self, state1: int, state2: int, as_dict: bool = False) -> Union[Dict, List]:
        """Gives information about the frames at which the hops occur.

        Args:
            - state1 (int):
            - state2 (int):
            - as_dict (bool): If set as True returns a Dict. Defaults to False.

        Returns:
            hop_data (Dict) or (List): hop frames as dict or list.
        """
        if as_dict:
            hop_data = {}
            for traj in self.trajs:
                hop_data[f"traj_{traj.traj_no}"] = traj.hop_frames_actual(state1, state2)
        else:
            hop_data = []
            for traj in self.trajs:
                hop_data.append(traj.hop_frames_actual(state1, state2))

        return hop_data
