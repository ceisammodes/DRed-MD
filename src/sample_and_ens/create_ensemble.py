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
This script creates an <Ensemble> object made of trajectories (<Trajectory>
object) themselves composed of frames for each timestep (<Frame> object).
  
  Morgane Vacher, Vincent Delmas, Isabella Merritt (2023)

*******************************************************************************
"""

import sys
import os
from typing import List
import argparse

""" ----------------------------------------------------------------------------------------------------- """

# Add the /src directory to $PATH
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import scripts.class_TSH as TSH
from scripts.utilities import pickle_save

PROGS = {1: "SHARC", 2: "NX", 3: "MOLCAS"}

""" ----------------------------------------------------------------------------------------------------- """


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='create_ensemble.py',
        description='Creates an ensemble of all trajectories present in the current directory',
        epilog='Don\'t forget to drink water!'
    )

    parser.add_argument('-o', '--output', type=str, default='ensemble.pickle',
                        help='name of the output file (default: ensemble.pickle)')

    return parser.parse_args()


""" ----------------------------------------------------------------------------------------------------- """


def check_timestep_sharc(trajs_idx: List):
    """Check that the SHARC timestep is consistent between all trajectories.
    
    Args:
        trajs_idx (List): indeces of the trajectory folders.
    
    Returns:
        None. It checks the time step among the various trajectories.

    Raises:
        ValueError if the time step is not consistent.
    """
    # Get timestep from first trajectory
    # with open(f"TRAJ{trajs_idx[0]}/output.lis") as fp:  # Previous directory nomenclature
    with open("TRAJ_%s/output.lis" % str(trajs_idx[0]).zfill(4)) as fp:  # Directory nomenclature
        output = fp.read().splitlines()

    # Main timestep
    timestep = float(output[6].split()[1])

    # Check timestep consistency
    for idx in trajs_idx:
        # with open(f"TRAJ{idx}/output.lis") as fp:  # Previous directory nomenclature
        with open("TRAJ_%s/output.lis" % str(idx).zfill(4)) as fp:  # Directory nomenclature
            output = fp.read().splitlines()

        traj_timestep = float(output[6].split()[1])
        if traj_timestep != timestep:
            raise ValueError(f"TRAJ{idx} timestep ({traj_timestep}) =/= TRAJ{trajs_idx[0]} timestep ({timestep})")


def check_timestep_molcas(trajs_idx: List, file_keys: List):
    """Check that the MOLCAS timestep is consistent between all trajectories.
    
    Args:
        trajs_idx (List): indeces of the trajectory folders.
        file_keys (List): the root of the Molcas input file.
    
    Returns:
        None. It checks the time step among the various trajectories.

    Raises:
        ValueError if the time step is not consistent.
    """
    with open(f"TRAJ{trajs_idx[0]}/{file_keys[-1]}_{trajs_idx[0]}.md.energies") as fp:
        data = fp.read().splitlines()

    # Main timestep
    timestep = float(data[2].split()[0])

    for idx in trajs_idx:
        with open(f"TRAJ{idx}/{file_keys[-1]}_{idx}.md.energies") as fp:
            data = fp.read().splitlines()

        traj_timestep = float(data[2].split()[0])
        if traj_timestep != timestep:
            raise ValueError(f"TRAJ{idx} timestep ({traj_timestep}) =/= TRAJ{trajs_idx[0]} timestep ({timestep})")


def check_timestep_nx(trajs_idx: List):
    """Check that the NX timestep is consistent between all trajectories.
    
    Args:
        trajs_idx (List): indeces of the trajectory folders.
    
    Returns:
        None. It checks the time step among the various trajectories.

    Raises:
        ValueError if the time step is not consistent.
    """
    with open(f"TRAJ{trajs_idx[0]}/RESULTS/en.dat") as fp:
        data = fp.read().splitlines()

    # Main timestep
    timestep = float(data[1].split()[0])

    for idx in trajs_idx:
        with open(f"TRAJ{idx}/RESULTS/en.dat") as fp:
            data = fp.read().splitlines()

        traj_timestep = float(data[1].split()[0])
        if traj_timestep != timestep:
            raise ValueError(f"TRAJ{idx} timestep ({traj_timestep}) =/= TRAJ{trajs_idx[0]} timestep ({timestep})")


if __name__ == "__main__":
    args = parse_arguments()

    print("\n=====================================================================================================")
    print("                  TSH Ensemble Generation Program (collect data for Post-Processing)                 ")
    print("=====================================================================================================")

    # Select program
    print("\n--- SELECT A PROGRAM --------------")
    print("1: SHARC")
    print("2: NX")
    print("3: MOLCAS")
    prog_nb = int(input("Choose your program: "))
    print("-----------------------------------")

    if PROGS[prog_nb].lower() not in TSH.VALID_PROGRAMS:
        raise Exception(f"Choice not recognized, options are: {', '.join(TSH.VALID_PROGRAMS)}")

    # Select <TRAJ> and skipped
    all_trajs = sum("TRAJ" in f for f in os.listdir())
    print(f"\nNumber of 'TRAJ' folders located: {all_trajs}")
    nb_trajs = input("Total trajectories (including failed ones) [default: all]: ")
    nb_trajs = all_trajs if nb_trajs == "" else int(nb_trajs)
    skipped_trajs = [int(t) for t in input("Trajectories to skip (space separated): ").split()]
    for s_t in skipped_trajs:
        if not 0 < s_t <= nb_trajs:
            raise Exception(f"Invalid option for trajectory to skip (TRAJ{s_t}) - not in range of all trajectories.")
    try:
        max_time = input("Upper limit of timesteps [default: None]: ")
        max_time = None if max_time == "" else int(max_time)
    except TypeError:
        print("Invalid option for the number of timesteps, not setting a limit")
        max_time = None

    # Give <file_keys> if used with molcas
    file_keys = []
    if PROGS[prog_nb].lower() == "molcas":
        file_keys = input("Insert filekey(s) (name without <_traj_no>, space separated if list): ").split()

    # All trajectories indices except skipped ones (*** BEGIN AT 1 ***)
    trajs_idx = [t for t in range(1, nb_trajs + 1) if t not in skipped_trajs]

    # Check that <timestep> consistent between all trajectories
    if PROGS[prog_nb].lower() == "molcas":
        check_timestep_molcas(trajs_idx, file_keys)
    elif PROGS[prog_nb].lower() == "nx":
        check_timestep_nx(trajs_idx)
    elif PROGS[prog_nb].lower() == "sharc":
        check_timestep_sharc(trajs_idx)
    else:
        print("Program not recognized. Exiting.")
        exit()

    print("\n-----------------------------------------------------------------------------------------------------")
    print(f"Creating <Ensemble> for {nb_trajs} trajectories run using <{PROGS[prog_nb]}>")
    print(f"Excluding trajectories: {' '.join([str(t) for t in skipped_trajs])}")
    print("-----------------------------------------------------------------------------------------------------")

    expt = TSH.Ensemble(
        prog=PROGS[prog_nb],
        trajs_idx=trajs_idx,
        file_keys=file_keys,
        max_time=max_time,
    )

    # Save pickled experiment
    pickle_save(args.output, expt)
    print("=====================================================================================================\n")
