python -m pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

srun --pty --gres=gpu:1 --cpus-per-gpu=8 bash

sample_image=/home/c3-0/datasets/LTCC/LTCC_ReID/train/094_1_c9_015923.png
# cp $sample_image ./
# cp /squash/siddiqui-schp-logs/LTCC/train/013_2_c12_006327.png ./

conda activate pathak2 

python SAM.py --filepath "$sample_image" --savedir "outputs/silhouettes" --image \
                --masks "person" \
                --masks "shirt" \
                --masks "pant" \
                --prompts "bbox" 
                