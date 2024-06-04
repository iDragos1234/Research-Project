#!/bin/bash

#SBATCH --job-name="dicom_preprocessing"
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=preproc_bg.out
#SBATCH --error=error_bg.out


module load 2023r1
module load python/3.9.8
module load py-pip
module load py-numpy
module load py-scikit-learn
module load py-tqdm


python -m pip install --user -r /scratch/dileana/research-project/requirements.txt


srun python /scratch/dileana/research-project/preprocessor_module/main.py \
            --input-dir "/scratch/dileana/oa2024/shared/data" \
            --output-hdf5 "/scratch/dileana/research-project/all_bg.h5" \
            --percentile-normalization 5 95 \
            --target-pixel-spacing 0.9 0.9 \
            --target-pixel-array-shape 512 512 \
            --include_background-mask \
            --verbose
