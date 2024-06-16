#!/bin/bash

#SBATCH --job-name="model_training"
#SBATCH --partition=gpu
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
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
module load py-torch
module load py-scikit-learn
module load py-tensorboard
module load py-tqdm


python -m pip install --user -r /scratch/dileana/research-project/requirements.txt


srun python /scratch/dileana/research-project/trainer_module/main.py \
            --input-hdf5 /scratch/dileana/research-project/preprocessed_data/all_no_bg.h5 \
            --input-data-split-csv /scratch/dileana/research-project/data_split.csv \
            --output-model-state-filepath /scratch/dileana/research-project/results/best_model.pth \
            --output-stats-dir /scratch/dileana/research-project/results \
            --model 2 \
            --device cuda \
            --max-epochs 100 \
            --batch-size 100 \
            --num-workers 0 \
            --validation-interval 1 \
            --seed 42 \
            --verbose
