#!/bin/bash

#SBATCH --job-name="calculate_jsw"
#SBATCH --partition=compute
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-eemcs-courses-cse3000


module load 2023r1
module load openmpi
module load cuda
module load python/3.9.8
module load py-pip
module load py-h5py
module load py-numpy
# module load py-torch
module load py-scikit-learn
module load py-tensorboard
module load py-tqdm


python -m pip install --user -r /scratch/dileana/research-project/requirements.txt


srun python /scratch/dileana/research-project/jsw_calculator_module/main.py \
            --input-hdf5 ./all_with_bg.h5 \
            --input-data-split-csv ./data_split.csv \
            --input-model-state-filepath ./training_v2_13-06-2024_22-00/best_metric_model.pth \
            --output-dir ./jsw_results \
            --model 2 \
            --seed 42 \
            --verbose
