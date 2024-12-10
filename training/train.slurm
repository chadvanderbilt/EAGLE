#!/bin/bash
#SBATCH --job-name=train_EGFR
#SBATCH --output=train_EGFR.out
#SBATCH --ntasks=24
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00  # Adjust the time as needed
#SBATCH --partition=gpu   # Adjust partition name as per your cluster setup

SCRIPT="train.py --tilesize 224 --k_per_gpu 96 --optimizer adamw --lr 1e-06 --warmup_epochs 2 --nepochs 20 --workers 10 --save_freq 2 --use_amp 1 --target EGFR_KD --drop 0.5 --pos_weight 0.7"

# Calculate the number of nodes and tasks
NHOST=$SLURM_JOB_NUM_NODES
NPPN=$SLURM_NTASKS_PER_NODE

# PyTorch distributed training command
PCOMMAND="torchrun"
PCOMMAND="$PCOMMAND --nproc_per_node=$NPPN"
PCOMMAND="$PCOMMAND --nnodes=$NHOST"
PCOMMAND="$PCOMMAND --rdzv_id=200"
PCOMMAND="$PCOMMAND --rdzv_backend=c10d"
PCOMMAND="$PCOMMAND --rdzv_endpoint=$SLURM_NODELIST:29400"
PCOMMAND="$PCOMMAND $SCRIPT"

echo "$PCOMMAND"

# Activate conda environment and run the command
module load cuda/11.8  # Load CUDA module (adjust version as necessary)
conda activate H100NVL
srun --ntasks=$SLURM_NTASKS $PCOMMAND
