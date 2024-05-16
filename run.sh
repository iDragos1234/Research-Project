#!/bin/bash

#SBATCH --job-name="Py_pi"
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=preproc.out
#SBATCH --error=error.out


module load 2023r1
module load openmpi
module load python/3.9.8
module load py-mpi4py
module load py-numpy
module load py-torch
module load py-pip


python -r pip install --user tqdm 
python -r pip install --user matplotlib
python -r pip install --user h5py
python -r pip install --user hdf5plugin
python -r pip install --user circle_fit
python -r pip install --user monai-weekly


srun python ./dicom_preprocessor/dicom_preprocessor.py \
               --input "./data" \
               --output "./output.h5" \
               --target-pixel-spacing 1 \
               --limit 5
