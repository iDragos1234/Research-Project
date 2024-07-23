#!/bin/bash

#SBATCH --job-name="data preprocessing"
#SBATCH --partition=compute
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-eemcs-courses-cse3000


module load 2023r1
module load python/3.9.8
module load py-pip
module load py-numpy
module load py-scikit-learn
module load py-tqdm


python -m pip install --user -r /absolute/path/to/research-project/requirements.txt


srun python /absolute/path/to/research-project/preprocessor_module/main.py \
            --input-dir "/absolute/path/to/research-project/data" \
            --output-hdf5 "absolute/path/to/research-project/all_bg.h5" \
            --percentile-normalization 5 95 \
            --target-pixel-spacing 0.9 0.9 \
            --target-pixel-array-shape 512 512 \
            --include-background-mask \
            --verbose
