#!/bin/bash

#SBATCH --job-name="train_unet_v1"
#SBATCH --partition=compute
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=train_unet_v1.out
#SBATCH --error=error_train_unet_v1.out


module load 2023r1
module load openmpi/4.1.4
module load cuda/12.1
module load python/3.9.8
module load py-pip/22.2.2
module load py-h5py/3.7.0
module load py-mpi4py/3.1.4
module load py-numpy/1.22.4
module load py-torch/1.12.1
module load py-scikit-learn/1.1.3
module load py-tensorboard/2.8.0
module load py-tqdm/4.64.1


python -m pip install --user -r /scratch/dileana/research-project/requirements.txt


srun python /scratch/dileana/research-project/trainer_module/main.py \
            --input-hdf5 /scratch/dileana/research-project/preprocessed_data/all_no_bg.h5 \
            --input-data-split-csv /scratch/dileana/research-project/data_split.csv \
            --output-model-dir /scratch/dileana/research-project/results \
            --device cpu \
            --learning-rate 1e-3 \
            --weight-decay 0 \
            --max-epochs 100 \
            --batch-size 100 \
            --num-workers 0 \
            --validation-interval 1 \
            --seed 42 \
            --verbose
