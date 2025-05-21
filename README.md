# ***Dynamics in Reduced dimensionality***

<img src="/docs/logo.png" width="500" height="500">

# In this repository
- **docs**: This directory contains the detailed documentation about the collection of programs used for performing surface hopping dynamics in reduced dimensionality.
- **src**: This directory contains all the needed modules to perform dynamics in reduced dimensionality.
  - **sample_and_ens**: This directory contains Python scripts for creating `ensemble` objects from surface hopping trajectories, obtained from OpenMolcas, SHARC or NewtonX, in the form of `.pickle` files. Further details are provided below. The `ensemble` object contains the reference full dimensional data set and it can be seen as the training set on which Principal Component Analysis (PCA) or Normal Mode Variance (NMV) will be performed.
  - **scripts**: This directory contains all the necessary dependencies for basically all the other scripts, the functions in `utility` module, and the main classes in `class_TSH`.
  - **scripts_reduce_dyn**: This directory contains Python scripts dedicated to the reduction of the initial conditions in order to start the dynamics in reduced dimensionality.
  - **transformers**: This directory contains the Python script dedicated to the reduction of dimensions using OpenMolcas.
- **templates**: This directory contains some template of OpenMolcas input files and `slurm` submission files for the Jean Zay supercomputer (installed at IDRIS, a national computing centre for the CNRS).
- Some examples of dynamics run in reduced dymensionality are available at `https://uncloud.univ-nantes.fr/index.php/apps/files/files/1958632381?dir=/ATTOP-DATA/TESTS_RED_DIM` and accessible to everyone via `wget -O tests.tar.gz https://uncloud.univ-nantes.fr/index.php/s/5i2AoH2WtfGWsR3/download/tests_210525.tar.gz`

## Table of contents:

- [Prerequisites](#Prerequisites)
  - [Installation](#Installation)
- [Presentation of the functionalities](#Presentation-of-the-functionalities)
  - [Create Ensemble](#Create-Ensemble)
  - [Create PCA and NMV containers](#Create-PCA-and-NMV containers)
  - [MD in reduced dimensionality](#MD-in-reduced dimensionality)
- [Example of use](#Example-of-use)
- [About us](#About-us)
- [Acknowledgements](#Acknowledgements)

# Prerequisites

To take advantage of all the modules present in this repository, consider to install all the the dependencies.
- Python 3.7+ (Developed and tested with python 3.9)
- pip3

## Installation

### Via pip
The installation of the required packages can be done, for instance, via `pip` or `pip3` package manager:
```
pip install -r requirements.txt
```

### Via Conda
Alternatively, installation can be done using then [CONDA](https://docs.conda.io/projects/conda/en/latest/index.html) virtual environment:
```
conda create -n your_env_name python=3.9
conda activate your_env_name
conda install pip
pip install -r requirements.txt
```

### Via Docker
[Docker](https://www.docker.com/) can be used to use DRed-MD in an isolated environment. Go into the DRed-MD folder and create an `image` named `<User>/dred-md`:
```
docker build -t <User>/dred-md .
```

Then, create a `container` named `test_run`:

```
docker run --user $UID:$GID  -it --rm -v $PWD/:/dred-md/ --name test_run <User>/dred-md
```

The DRed-MD directory has has been mounted inside the container. Any changes, such as copying a directory `data` into the DRed-MD directory on your machine, will be accessible within the container under `/dred-md/data`.

# Presentation of the functionalities

## Create Ensemble

### Create ens - `src/sample_and_ens/create_ensemble.py`
After running Tully Surface Hopping (TSH) trajectories the `create_ensemble.py` script can be used to create an `ensemble` object in the form of a pickle binary file. The `Ensemble` class contains `Trajectory` class for each trajectory in the selected folder, and each Trajectory contains `Frame` classes which 
represents the molecular data at each timestep.

***_Example_***. Given 100 trajectories of 600 steps, after running the `create_ensemble.py`, the binary file will contain an `ensemble` object containing 100 `trajectory` object and each trajectory object will contain 600 `frame` objects which will contain data like:
- geometry
- velocities
- kinetic energy
- potential energy
- total energy
- eigenvalues
- electronic populations
- active state
- _etc_

## Create PCA and NMV containers

### Create PCA - `src/scripts_reduce_dyn/create_PCA.py`
After the `ensemble.pickle` file is created, it can be read with `create_PCA.py` and PCA can be performed on the full dimensional data set, i.e., the reference data set. The number of principal components (PCs) can be selected directly inside the script. After running `create_PCA.py`, the PCs and all the required information to perform dynamics in reduced dimensionality are stored in a new `.pickle` file that will be read from OpenMolcas during the dynamics.

### NM Variance - `src/scripts_reduce_dyn/nm_variance.py`
After the `ensemble.pickle` file is created, it can be read with `nm_variance.py` that computes the NMV and removes the selected NM with low variance associated. After running `nm_variance.py` the NMs to be included and all the information to perform dynamics in reduced dimensionality are stored in a new `.pickle` file that will be read from OpenMolcas during the dynamics.

## MD in reduced dimensionality
Finally, after the `.pickle` file containg information regarding the PCA or the NMV is available, MD in reduced dimensionality can be performed using OpenMolcas and the `src/transformers/transformer.py` module. Both the `.pickle` file and the `transformer.py` should be present in the folder in which the MD is run. An example of input file for running in reduced dimensionality is given in `templates/molcas_dyn_input.template`.

# ***trans***-to-***cis*** isomerisation of AZM in reduced dimensionality: a test case

![Alt text](/docs/1234.png)

# Example of use

We provide, as a minimal example, the reduced dimensional dynamics of trans-AZM upon excitation to the S1 electronic state and the procedure followed to run in reduced dimensionality. Our intention with this example is not to show quantitative results about the isomerisation process of trans-AZM but rather to illustrate the simple but completely general procedure that can be followed to run simulations in reduced dimensionality within the present package.

The first step is cloning the repository, for example with `git clone https://gitlab.univ-nantes.fr/modes/attop/DRed-MD.git`. In the new `DRed-MD` repository folder the data for running this test are not present but can be downloaded, as indicated before, via `wget -O tests.tar.gz https://uncloud.univ-nantes.fr/index.php/s/5i2AoH2WtfGWsR3/download/tests_210525.tar.gz` and extracted with `tar -xvzf tests.tar.gz`. To summarise,

```
git clone https://gitlab.univ-nantes.fr/modes/public/attop/DRed-MD.git
```

```
cd DRed-MD
mkdir tests
cd tests
wget -O tests.tar.gz https://uncloud.univ-nantes.fr/index.php/s/5i2AoH2WtfGWsR3/download/tests_210525.tar.gz
tar -xvf tests.tar.gz
```
After running these commands, finally you should have access to the `tests_210525/` folder.

Actually at this step you can launch your docker container or activate your python virtual environment (see [Installation](#Installation))

We assume that a set of full dimensionality dynamics is available because it represents the training set necessary to perform the either the PCA or MNV analyses. This minimal illustrative set comprising nine full dimensionality trajectories (`TRAJ1/`, `TRAJ2/`, ..., `TRAJ9/`) can be found in the `tests_210525/trans_AZM/reference_ensemble/` folder.

The first step is the creation of the `ensemble.pickle` file that contains the information about the reference (training) set with `create_ensemble.py`. Enter into the `reference_ensemble/` folder and make sure that the `src/script/` folder, which contains the `create_ensemble.py` dependencies, is in the current working directory. The `script` folder is initialised to be a package containing all the necessary dependencies. Then, with:

- python create_ensemble.py

  ```
  =====================================================================================================
                    TSH Ensemble Generation Program (collect data for Post-Processing)                 
  =====================================================================================================

  --- SELECT A PROGRAM --------------
  1: SHARC
  2: NX
  3: MOLCAS
  Choose your program: 
  ```

  you are asked which program was used for the full dimensional simulations, with compatibility with SHARC, NX, and OpenMolcas. In this case, select `3`. Then you are asked about the Total trajectories, Trajectories to skip, Upper limit of timesteps. For this example we will not dive into these functionalities and we can press `Enter` until **Insert filekeys** . Finally, we have to Insert filekey(s) which should be followed by the root of the molcas input name. In this case is just `molcas_input` (just check in one of the nine folders to confirm it).

  ```
  Number of 'TRAJ' folders located: 9
  Total trajectories (including failed ones) [default: all]: 
  Trajectories to skip (space separated): 
  Upper limit of timesteps [default: None]: 
  Insert filekey(s) (name without <_traj_no>, space separated if list): molcas_input
  ```
  
  After the last entry, some information is printed at screen:

  ```
  Info             | Value(s)               |
  -------------------------------------------
  Prog. used       | Molcas                 |
  En. threshold    | 0.5 eV                 |
  Nb trajs         | 9                      |
  Nb trajs removed | 0                      |
  Trajs removed    | []                     |
  Timestep         | 0.48377685080000005 fs |
  Nb steps         | 579                    |
  Total sim. time  | 280.10679661320006 fs  |

  Creating pickle file...
  Pickle saved: ensemble.pickle
  ```

The `ensemble.pickle` file is produced and contains information about the reference set (see Create Ensemble section for more detail about the information in the binary). The latter binary file, along with the `.output` file of the frequency calculation on the equilibrium geometry of the molecule, can be used by `create_PCA.py` or `nm_variance.py` to obtain the `.pickle` file for running in reduced dimensionality with the selected number of PCs or NMs, respectively. Thus, the next step is:

- python create_PCA.py or python nm_variance.py
  
  E.g., with the `create_PCA.py` script it is possible to select the number(s) of dimensions to keep passing them as a list interactively:
  ```
  python create_PCA.py -i ensemble.pickle -f freq.output -ft mckly
  ```
  ```
  Loading pickle...
  Data loaded!

  Found 10 atoms in the OpenMolcas frequency files.
  Found 24 normal modes in the OpenMolcas frequency files.
  Enter the number of principal components to keep.
  You can specify multiple integers to generate multiple files.
  Example: 2 4 31

  > 18
  ```
  In the case of PCA, a file named `PCA_k_comp_nm.pickle`, with `k` the number of selected PCs, is obtained. In the previous example, `k = 18`. It contains the information about the PCA done on the training set. In the case of NMV, a file named `container_var_k_dim_nm.pickle` with `k` the number of selected NMs ordered by variance is obtained.

All the ingredients necessary to run in reduced dimensionality are ready! In the directory: `tests/tests_210525/trans_AZM/DRed-MD_trajs/`, some trajectory folders are already prepared and named `TRAJ1/`, `TRAJ2/`, ..., `TRAJ9/`. It is possible to find also the `PCA_18_comp_nm.pickle` file copied from the `reference_ensemble/` folder.
Note that in each folder, in order to run in reduced dimensionality, it is necessary to have:

- the PCA or NMV `.pickle` file
- `transformer.py` module (part of the OpenMolcas suite) that can be found also in this repository in `src/transformers/`

in addition to the initial conditions and the input file. Note in the `.input` file the `red` keyword in the `&DYNAMIX` module. The arborescence of the folder in which the reduced dimensionality dynamics will be performed should be:
```bash
TRAJX/
├── geom_X.xyz
├── velocity_X.xyz
├── molcas_input_X.dyn.input
├── PCA_k_comp_nm.pickle
└── transformer.py
```
This folder can be transferred on the machine of choice with the compiled version of OpenMolcas that offers the possibility to run in reduced dimensionality.

***For the beta-testers:
I used Jean-Zay and the OpenMolcas version in /linkhome/rech/gencei01/uqv47eu/soft/openmolcas/build-dev-gl2/.
The submission script can be found in templates/sub_array.slurm in this repository or as follows:***
```
#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -p cpu_p1
#SBATCH -J "DRed-MD-test-AZM"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --threads-per-core=2
#SBATCH --cpus-per-task=4
#SBATCH --qos=qos_cpu-t3
#SBATCH --hint=nomultithread
#SBATCH --array=[1-9]


##################### MOLCAS #########################

module purge

export MOLCAS=/linkhome/rech/gencei01/uqv47eu/soft/openmolcas/build-dev-gl2

module load anaconda-py3/2021.05
module load cmake/3.21.3
module load intel-compilers/19.1.3
module load intel-mkl/2020.4
module load hdf5/1.12.0

source /linkhome/rech/gencei01/uqv47eu/.Molcas/molcasrc

export WorkDir=$JOBSCRATCH
export MOLCAS_NPROCS=$SLURM_NTASKS
export MOLCAS_MEM=$SLURM_MEM_PER_CPU

export NAME="molcas_input_${SLURM_ARRAY_TASK_ID}"

cd TRAJ${SLURM_ARRAY_TASK_ID}

/linkhome/rech/gencei01/uqv47eu/soft/openmolcas/build-dev-gl2/pymolcas $NAME.input >& $NAME.output
```

Finally, all is ready, happy dynamix!

# About Us

[The ATTOP team](https://morganevacher.wordpress.com/attop-project-members/)

# Acknowledgements

Part of the calculations in the project were performed on the Jean Zay supercomputer (http://www.idris.fr/eng/jean-zay/) and the GLiCID cluster (https://doi.org/10.60487/glicid)