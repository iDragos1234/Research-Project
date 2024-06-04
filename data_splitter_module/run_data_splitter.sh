#!/bin/bash

#SBATCH --job-name="data_split"
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=data_split.out
#SBATCH --error=error_data_split.out


module load 2023r1
module load python/3.9.8
module load py-pip


python -m pip install --user -r /scratch/dileana/research-project/requirements.txt


srun python /scratch/dileana/research-project/data_splitter_module/main.py \
            --input-hdf5 /scratch/dileana/research-project/preprocessed_data/all_no_bg.h5 \
            --output-csv /scratch/dileana/research-project/data_split.csv \
            --ratios 0.8 0.1 0.1 \
            --seed 42 \
            --verbose