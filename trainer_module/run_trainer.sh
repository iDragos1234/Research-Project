#!/bin/bash

#SBATCH --job-name="train_unet_v1"
#SBATCH --partition=compute
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=train_unet_v1.out
#SBATCH --error=error_train_unet_v1.out


module load 2023r1
module load openmp
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
            --output-model-dir /scratch/dileana/research-project/results \
            --model 2 \
            --device cpu \
            --learning-rate 1e-3 \
            --weight-decay 1e-5 \
            --max-epochs 100 \
            --batch-size 100 \
            --num-workers 0 \
            --validation-interval 1 \
            --seed 42 \
            --verbose
