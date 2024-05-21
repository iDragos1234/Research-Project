#!/bin/bash

#SBATCH --job-name="UNet_training"
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=unet_train.out
#SBATCH --error=error.out


module load 2023r1
module load openmpi
module load cuda
module load python/3.9.8
module load py-mpi4py
module load py-pip
module load py-numpy
module load py-scikit-learn
module load py-tqdm


python -m pip install --user -r /scratch/dileana/research-project/requirements.txt


previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun python /scratch/dileana/research-project/unet_trainer/model.py \
            --input /scratch/dileana/research-project/CHECK_1000.h5 \
            --model-dir /scratch/dileana/research-project/unet_trainer/results \
            --test-size 0.1 \
            --validation-size 0.1 \
            --train-size 0.8 \
            --device cuda \
            --learning-rate 1e-3 \
            --weight-decay 0 \
            --max-epochs 100 \
            --batch-size 20 \
            --num-workers 0 \
            --validation-interval 1 \
            --verbose

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
