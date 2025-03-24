#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -p cpu_p1
#SBATCH -J "TrAZB_full"
#SBATCH -e /dev/null
#SBATCH -o /dev/null
#SBATCH --ntasks=1
#SBATCH --threads-per-core=2
#SBATCH --cpus-per-task=4
#SBATCH --qos=qos_cpu-t3
#SBATCH --hint=nomultithread
#SBATCH --array=1-100


##################### MOLCAS #########################


module purge

# export MOLCAS=/gpfswork/rech/ucx/uzx44ef/soft/Openmolcas/build-git2
# module load intel-compilers/19.1.3
# module load intel-mkl/19.0.4
# module load hdf5/1.10.5
# module load cmake/3.18.0
# module load anaconda-py3/2020.11

# source /linkhome/rech/gencei01/uzx44ef/.Molcas/molcasrc

export MOLCAS=/linkhome/rech/gencei01/uqv47eu/soft/openmolcas/build-dev
module load anaconda-py3/2021.05
module load cmake/3.21.3
module load intel-compilers/19.1.3
module load intel-mkl/2020.4
module load hdf5/1.12.0

source /linkhome/rech/gencei01/uqv47eu/.Molcas/molcasrc

export WorkDir=$JOBSCRATCH
export MOLCAS_NPROCS=$SLURM_NTASKS
export MOLCAS_MEM=$SLURM_MEM_PER_CPU

export NAME="TrAZB_full_${SLURM_ARRAY_TASK_ID}"

cd TRAJ${SLURM_ARRAY_TASK_ID}

pymolcas $NAME.input >& $NAME.output
