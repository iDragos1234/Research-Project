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
module load cuda
module load python/3.9.8
module load py-mpi4py
module load py-pip
module load py-numpy
module load py-torch
module load py-scikit-learn
module load py-tqdm


python -m pip install --user -r /scratch/dileana/research-project/requirements.txt


srun python /scratch/dileana/research-project/dicom_preprocessor/dicom_preprocessor.py \
            --input "/scratch/dileana/oa2024/shared/data" \
            --output "/scratch/dileana/research-project/preprocessed_data.h5" \
            --target-pixel-spacing 1 \
            --limit 5
