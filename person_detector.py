import json, os
import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import torch
import magic 

from mmcv.fileio import FileClient
import decord
import io as inot

def convert_to_builtin_type(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj)))

def filedata(filepath):
    return magic.from_file(filepath, mime=True)

def save_json(data, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, default=convert_to_builtin_type)

    if args.verbose:
        print(f'Pose data saved to {json_path}')

def load_video(video_path):
    file_obj = inot.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=1)
    container = [img.asnumpy() for img in container]
    return container

def process_image(image_path, json_dir, model):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width, height, _ = img.shape
    results = model([img], size=640)

    if len(results.xyxy[0]) == 0:
        print(image_path)
        return 0
    x0, y0, x1, y1, _, _ = results.xyxy[0][0].cpu().numpy().astype(int)

    data = {}
    data[0] = {
        'person_bb': [[x0, y0, x1, y1]]
    }

    info = {
        'width': width,
        'height': height,
        'pose_data': data
    }
    json_path = os.path.join(json_dir, Path(image_path).stem+".json")
    save_json(info, json_path)
    return info

def process_video(video_path, json_dir, model):
    imgs = load_video(video_path)
    data = {}
    for i, img in enumerate(imgs):
        width, height, _ = img.shape
        results = model([img], size=640)

        if len(results.xyxy[0]) == 0:
            print(video_path)
            return 0
        x0, y0, x1, y1, _, _ = results.xyxy[0][0].cpu().numpy().astype(int)

        data[i] = {
            'person_bb': [[x0, y0, x1, y1]]
        }

    info = {
        'width': width,
        'height': height,
        'pose_data': data
    }
    json_path = os.path.join(json_dir, Path(video_path).stem+".json")
    save_json(info, json_path)
    return info

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Segment Anything Model Person and Clothes Silhouettes")

    parser.add_argument("--filepath",  help="path to input video file", type=str, required=True)
    parser.add_argument("--jsonsavedir",  help="path where to save the json file", type=str, required=True)
    parser.add_argument("--verbose",  help="default: None, shows the output json savepath", action='store_true')

    parser.add_argument('--confidence', type=int, default=0.8)

    parser.add_argument('--scale_torso_width', type=int, default=1.5)
    parser.add_argument('--scale_torso_height', type=int, default=1.3)

    parser.add_argument('--scale_pants_width', type=int, default=2)
    parser.add_argument('--scale_pants_height', type=int, default=1.3)

    parser.add_argument('--scale_person_width', type=int, default=1.5)    
    parser.add_argument('--scale_person_height', type=int, default=1.4)

    args = parser.parse_args()

    # filepath = f"DatasetB-1/video/{filename}.avi"
    # jsonsavepath = f"casia/jsons/{filename}.json"

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    if 'image' in filedata(args.filepath):
        process_image(args.filepath, args.jsonsavedir, model)
    if 'video' in filedata(args.filepath):
        process_video(args.filepath, args.jsonsavedir, model)

    # folderpath = "/home/c3-0/datasets/LTCC/LTCC_ReID/query/"
    # json_root = "outputs/LTCC/jsons/query"
    # for img_path in os.listdir(folderpath):
    #     img_path = os.path.join(folderpath, img_path)
        
    #     process_image(img_path, json_root, model)
    
    # process_video(args.filepath, args.jsonsavedir, model)


# python person_detector.py --filepath "/home/c3-0/datasets/LTCC/LTCC_ReID/train/094_1_c9_015923.png" --jsonsavedir "outputs/jsons"