#!/bin/bash 

# ======== CPU Config ======== 
#SBATCH -c 3
# SBATCH --mem=24G

# ======== GPU Config ======== 
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem11

# ======== Slurm config ======== 
#SBATCH --job-name=JS # job name
#SBATCH -o slurm_outputs/createjsons_LTCC_%j.out

module add anaconda3/2019.03-1
module add cuda/12.1

# conda activate gsam

dir="/home/c3-0/datasets/LTCC/LTCC_ReID/train"

for item in "$dir"*; do    
    python pose.py --filepath "$item" \
                --jsonsavedir "outputs/jsons"
done