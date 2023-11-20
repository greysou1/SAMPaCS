#!/bin/bash 

# ======== CPU Config ======== 
#SBATCH -c 8
#SBATCH --mem-per-cpu=12G

# ======== GPU Config ======== 
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C '(ampere)'

# ======== Slurm config ======== 
#SBATCH --job-name=sils # job name
#SBATCH -o slurm_outputs/sils_%j.out

module add anaconda3/2019.03-1
module add cuda/12.1

# conda activate gsam

dir="/home/c3-0/datasets/LTCC/LTCC_ReID/train/"

for item in "$dir"*; do    
    python person_detector.py --filepath "$item" \
                --jsonsavedir "outputs/jsons"
done

for item in "$dir"*; do
    filename=$(basename "$item" | sed 's/\.[^.]*$//')
    python SAM.py --filepath "$item" \
                --jsonpath "outputs/jsons/$filename.json" \
                --savedir "outputs/silhouettes" \
                --masks "person" \
                --masks "shirt" \
                --masks "pant" \
                --prompts "bbox"
done 