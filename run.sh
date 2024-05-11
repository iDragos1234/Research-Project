#!/bin/bash

#SBATCH --job-name="Py_pi"
#SBATCH --time=00:10:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-mpi4py

srun python test2.py > pi.log
