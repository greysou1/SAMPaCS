# SAMPaCS: Segment Anything Person and Clothes Silhouettes 

## Installation

#### Segment Anything
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```
or clone the repository locally and install with
```bash
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

#### Libraries
```bash
pip install -r requirements.txt
```

## Model Checkpoints

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```python
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- `default or vit_h`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
- `vit_l`: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth).
- `vit_b`: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).
