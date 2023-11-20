import gc, os, json
from pathlib import Path
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from segment_anything import sam_model_registry
import moviepy.video.io.ImageSequenceClip

class SAM:
    def __init__(self, 
                 sam_checkpoint="sam_vit_h_4b8939.pth",
                 device='cuda',
                 batch_size=1,
                 mask_names=["person", "shirt", "pant"],
                 prompts=['bbox']):

        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self.batch_size = batch_size
        self.mask_names = mask_names
        self.prompts = prompts
        
        # load model
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(self.device)
    
    def load_json(self, json_filepath):
        with open(json_filepath, "r") as json_file:
            data = json.load(json_file)
        
        w, h = data["width"], data["height"]
        
        new_data = {}
        for key, item in data["pose_data"].items():
            new_data[int(key)] = {}
            # bounding boxes
            
            for bb_key in item.keys():
                if bb_key == 'prompt_point':
                    # the center point of torso to use as prompt point
                    new_data[int(key)]['prompt_point'] = item['prompt_point']
                elif bb_key == 'landmarks':
                    # (x, y) values of 22 landmarks that can be used as prompt points
                    new_data[int(key)]['landmarks'] = [[sublist['x'] * w, sublist['y'] * h] for sublist in item['landmarks']] # cd: coords_dict
                else:
                    new_data[int(key)][bb_key] = [item for sublist in item[bb_key] for item in sublist]

        return new_data
    
    def extract_video_masks(self, videopath, 
                            jsonpath=None, 
                            savedir=None, scaling_factor=1):
        """
        Extract clothing bounding boxes and masks from a video.

        Args:
            videopath (str): Path to the input video file.
            bboxes_jsonpath (str, optional): Path to the JSON file containing bounding box data.
                                            Defaults to None.
            savedir (str, optional): Directory path to save the extracted data.
                                    Defaults to None.
            prompt (str): can either be 'bbox' or 'point'

        Returns:
            None: The function saves the extracted clothing bounding boxes and masks as files.

        Notes:
            - The function extracts clothing information (bounding boxes and masks) from the frames of the input video.
            - The bounding box data is obtained from the `bboxes_jsonpath` file, if provided. Each frame's bounding box
            information is stored in a dictionary where keys represent the frame index and values are dictionaries
            with 'shirtbbox' and 'pantbbox' entries.
            - The function performs batched inference on the frames and extracts shirt and pant masks using a model.
            - The extracted clothing bounding boxes are saved as a JSON file in the directory specified by `savedir`.
            The JSON file structure contains the frame index as keys and lists of dictionaries with 'shirtbbox' and
            'pantbbox' entries.
            - The extracted shirt masks are saved as PNG images in the 'silhouettes-shirts' subdirectory within the
            appropriate directory structure specified by `savedir`.
            - The extracted pant masks are saved as PNG images in the 'silhouettes-pants' subdirectory within the
            appropriate directory structure specified by `savedir`.
        """

        # videoname = os.path.basename(videopath)
        videoname = Path(videopath).stem


        cap = cv2.VideoCapture(videopath)

        # 1. read all frames
        frame_i = 1
        frames = {}
        batched_input = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            desired_width = int(frame.shape[1] * scaling_factor)
            desired_height = int(frame.shape[0] * scaling_factor)
            frame = cv2.resize(frame, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
            frames[frame_i] = frame
            # cv2.imwrite("outputs/debug/resized_frame.jpg", frame)
            frame_i += 1
        
        # 2. load bboxes
        # personbboxes = self.load_bboxes(personbboxes_jsonpath)
        prompt_data = self.load_json(jsonpath)
        clothing_bboxes = {}  # clothing_bboxes = {"26": {"shirtbbox": list, "pantbbox": list}}
        clothing_masks = {}   # clothing_masks  = {"26": {"shirtmask": torch.Tensor, "pantmask": torch.Tensor}}
        # 4. save the data
        # a. save the bboxes: clothing_bboxes
        # bbox_savepath = os.path.join(savedir, "clothing-jsons")
        # if not os.path.exists(bbox_savepath): os.makedirs(bbox_savepath, exist_ok=True)
        # bbox_savepath = os.path.join(bbox_savepath, f"{videoname}.json")

        # b. save the masks:  clothing_masks
        mask_save_fol_paths = {}
        for item in self.mask_names:
            savepath = os.path.join(savedir, f"silhouettes-{item}")
            if not os.path.exists(savepath): os.makedirs(savepath, exist_ok=True)
            mask_save_fol_paths[item] = savepath

        # 3. batch inference
        for i in tqdm(range(0, len(prompt_data), self.batch_size), desc=videoname):
            batched_input = []
            # a. create batched input
            frame_indices = sorted(list(prompt_data.keys()), key=int)[i: min(i+self.batch_size, len(frames))]
            
            for frame_i in frame_indices:
                try:
                    person_bbox = prompt_data[frame_i]["person_bb"]
                    shirt_bbox = prompt_data[frame_i]["torso_bb"]
                    pant_bbox = prompt_data[frame_i]["pants_bb"]
                    prompt_point = prompt_data[frame_i]["prompt_point"]
                    pose_points = prompt_data[frame_i]["landmarks"]
                except:
                    continue
                
                input_data = {}
                input_data['image'] = torch.as_tensor(frames[frame_i], device=self.sam.device).permute(2, 0, 1).contiguous()
                input_data['original_size'] = frames[frame_i].shape[:2]
                input_data['input_label'] =  np.array([1]) # 1 is for foreground, 0 is for background

                if 'bbox' in self.prompts:
                    bboxes_t = {}
                    for item in self.mask_names:
                        if item == "person":
                            bboxes_t["person"] = [item*scaling_factor for item in person_bbox]
                        elif item == "shirt":
                            bboxes_t["shirt"] = [item*scaling_factor for item in shirt_bbox]
                        elif item == "pant":
                            bboxes_t["pant"] = [item*scaling_factor for item in pant_bbox]

                    clothing_bboxes[frame_i] = bboxes_t
                    input_data['boxes'] = torch.as_tensor(list(clothing_bboxes[frame_i].values()), device=self.sam.device)
                
                if 'point' in self.prompts:
                    if self.mask_names == ['person']:
                        prompt_point = [item*scaling_factor for item in prompt_point]
                        input_data['input_point'] =  np.array([prompt_point])
                    else:
                        print("ERROR! 'point' and 'pose' prompts can only be used for extracting person silhouettes!")
                        quit()
                elif 'pose' in self.prompts:
                    if self.mask_names == ['person']:
                        prompt_points = [item*scaling_factor for item in pose_points]
                        input_data['input_point'] =  np.array([prompt_points])
                    else:
                        print("ERROR! 'point' and 'pose' prompts can only be used for extracting person silhouettes!")
                        quit()

                batched_input.append(input_data)

            if len(batched_input) == 0: continue
            
            # b. inference batched input
            batch_output = self.sam(batched_input, multimask_output=True)
            gc.collect()
            
            # c. index batched output
            for frame_i, output in zip(frame_indices, batch_output):
                masks, _, _ = output.values()
                masks_dict = {}
                for mask_name, mask_item in zip(clothing_bboxes[frame_i].keys(), masks[:len(clothing_bboxes[frame_i])]):
                    masks_dict[mask_name] = mask_item

                clothing_masks[frame_i] = masks_dict
        
        self.save_clothing_data(videoname, clothing_bboxes, clothing_masks, mask_save_fol_paths)
    
    def get_prompt_points(self, g_image, num=10):
        # Read the binary image
        image = cv2.imread(g_image, cv2.IMREAD_GRAYSCALE)
        # Find the coordinates of white pixels
        coords = np.column_stack(np.where(image == 255))
        coords = [coords[a] for a in range(0, len(coords), len(coords)//num)]
        return coords
    
    def extract_image_masks(self, imagepath, 
                            jsonpath=None, 
                            savedir=None, scaling_factor=1,
                            prompt_sil=None,
                            padding= 10):
        
        imagename = Path(imagepath).stem

        # 1. read frames
        img = cv2.imread(imagepath) #Image.open(imagepath).convert('RGB')
        width, height, _ = img.shape

        # desired_width = 192
        # desired_height = 384 
        # desired_width = int(width * scaling_factor)
        # desired_height = int(height * scaling_factor)
        # img = img.resize((desired_width, desired_height))        

        # 2. load bboxes
        prompt_data = None
        if jsonpath:
            # jsonpath="outputs/jsons/001-bg-01-000.json"
            prompt_data = self.load_json(jsonpath)
            clothing_bboxes = {}  # clothing_bboxes = {"26": {"shirtbbox": list, "pantbbox": list}}
            clothing_masks = {}   # clothing_masks  = {"26": {"shirtmask": torch.Tensor, "pantmask": torch.Tensor}}
        
        # 4. save the data
        # b. save the masks:  clothing_masks
        mask_save_fol_paths = {}
        for item in self.mask_names:
            savepath = os.path.join(savedir, f"silhouettes-{item}")
            if not os.path.exists(savepath): os.makedirs(savepath, exist_ok=True)
            mask_save_fol_paths[item] = savepath

        # 3. batch inference
        # if prompt_data:
        person_bbox = prompt_data[0]["person_bb"] # [136, 38, 172, 161]
            # shirt_bbox = prompt_data[0]["torso_bb"] # [0, 144, 87, 288]
            # pant_bbox = prompt_data[0]["pants_bb"] # [144, 94, 164, 147]
            # prompt_point = prompt_data[0]["prompt_point"] # [154.5152, 83.71560000000001]
            # pose_points = prompt_data[0]["landmarks"] # ...
        # else:
        #     person_bbox = [0,0,desired_width, desired_height]
        #     prompt_point = desired_width / 2, desired_height / 2
            
        #     shirt_height_start = max(desired_height // 2  - padding, 0)
        #     pant_height_end = min(desired_height // 2 + padding, desired_height)

        #     shirt_bbox = [0,0, desired_width, pant_height_end]
        #     # shirt_bbox = [0,shirt_height_start, desired_width, desired_height]
        #     # pant_bbox = [0,0, desired_width, pant_height_end]
        #     pant_bbox =  [0,shirt_height_start, desired_width, desired_height]

        input_data = {}
        input_data['image'] = torch.as_tensor(img, device=self.sam.device).permute(2, 0, 1).contiguous()
        input_data['original_size'] = img.shape[:2]
        input_data['input_label'] =  np.array([1]) # 1 is for foreground, 0 is for background

        if prompt_sil:
            prompt_points = self.get_prompt_points(prompt_sil, num=50)
            # img_copy = img.copy()
            # for pt in prompt_points:
            #     x, y = pt
            #     cv2.circle(img_copy, (y, x), 1, (0, 255, 0), -1)
            #     cv2.imwrite("debug.png", img_copy)

            input_data['input_point'] =  prompt_points
        batched_input = []
        if 'bbox' in self.prompts:
            bboxes_t = {}
            for item in self.mask_names:
                if item == "person":
                    bboxes_t["person"] = [item*scaling_factor for item in person_bbox]
                elif item == "shirt":
                    bboxes_t["shirt"] = [item*scaling_factor for item in shirt_bbox]
                elif item == "pant":
                    bboxes_t["pant"] = [item*scaling_factor for item in pant_bbox]

            clothing_bboxes = bboxes_t
            input_data['boxes'] = torch.as_tensor(list(clothing_bboxes.values()), device=self.sam.device)        
        if 'point' in self.prompts:
            if self.mask_names == ['person']:
                prompt_point = [item*scaling_factor for item in prompt_point]
                input_data['input_point'] =  np.array([prompt_point])
            else:
                print("ERROR! 'point' and 'pose' prompts can only be used for extracting person silhouettes!")
                quit()
        elif 'pose' in self.prompts:
            if self.mask_names == ['person']:
                prompt_points = [item*scaling_factor for item in pose_points]
                input_data['input_point'] =  np.array([prompt_points])
            else:
                print("ERROR! 'point' and 'pose' prompts can only be used for extracting person silhouettes!")
                quit()

        # print(input_data.keys())
        batched_input.append(input_data)

        # b. inference batched input
        batch_output = self.sam(batched_input, multimask_output=False)
        gc.collect()
            
        # c. index batched output
        for output in batch_output:
            masks, _, _ = output.values()
            masks_dict = {}
            for mask_name, mask_item in zip(clothing_bboxes.keys(), masks[:len(clothing_bboxes)]):
                masks_dict[mask_name] = mask_item
            clothing_masks = masks_dict

        self.save_clothing_data_as_Image(imagename, clothing_bboxes, clothing_masks, mask_save_fol_paths)
    
    def save_clothing_data(self, videoname, clothing_bboxes, clothing_masks, mask_save_paths_vid):
        """
        Saves clothing bounding box data and clothing mask frames to specified file paths.

        Args:
            clothing_bboxes (dict): Dictionary containing clothing bounding box data.
                Format: {frame_index: {'label': bbox_coordinates}}
            clothing_masks (dict): Dictionary containing clothing mask frames.
                Format: {frame_index: {'label': mask_tensor}}
            savepaths (tuple): Tuple containing file paths for saving bounding box data and mask videos.
                Format: (bbox_savepath, mask_save_paths)
            - bbox_savepath (str): File path to save bounding box data in JSON format.
            - mask_save_paths (dict): Dictionary containing file paths for saving mask videos.
                Format: {'label': video_savepath}

        Returns:
            None
        """
        image_list = {}
        image_list["person"] = []
        image_list["shirt"] = []
        image_list["pant"] = []
        
        for index, masks in clothing_masks.items():
            for mask_item, mask in masks.items(): 
                mask_np = mask.cpu().numpy()[0].astype(bool).astype(int)

                mask_img = torch.zeros(mask_np.shape)
                mask_img[mask_np == 1] = 1
                mask_img = (mask_img * 255).byte()     # Convert to uint8
                mask_img = mask_img.numpy()            # Convert tensors to numpy arrays
                
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

                image_list[mask_item].append(mask_img)
                
        fps = 15
        logger = 'bar' if args.verbose else None

        for mask_name in self.mask_names:
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_list[mask_name], fps=fps)
            clip.write_videofile(os.path.join(mask_save_paths_vid[mask_name], f"{videoname}.mp4"), fps=fps, codec="libx264", logger=logger)

    def save_clothing_data_as_Image(self, imagename, clothing_bboxes, clothing_masks, mask_save_paths_vid):
        for index, masks in clothing_masks.items():
            mask_np = masks.cpu().numpy()[0].astype(bool).astype(int)

            mask_img = torch.zeros(mask_np.shape)
            mask_img[mask_np == 1] = 1
            mask_img = (mask_img * 255).byte()
            mask_img = mask_img.numpy()
            
            mask_img = Image.fromarray(mask_img)
            mask_img.save(os.path.join(mask_save_paths_vid[index], f"{imagename}.png"))
            
    

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Segment Anything Model Person and Clothes Silhouettes")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--filepath",  help="path to input video file", type=str, required=False)
    parser.add_argument("--jsonpath",  help="path to input json file, this file is generated using the pose.py script", type=str, default=None, required=False)
    parser.add_argument("--verbose",  help="default: None, use to show the MoviePy library's logger output, shows the output savepath", action='store_true')
    parser.add_argument("--savedir",  help="path to directory where the silhouettes are to be saved", type=str, required=True)
    parser.add_argument('--masks', default=['person'], help='specify all the masks that you want to generate', 
                        action='append')
    parser.add_argument('--prompts', default=['bbox'], help='specify how to prompt SAM model, NOTE: currently only support \
                                                                     generating person silhouettes only using "point","pose" ', 
                        choices=['bbox','point','pose'], action='append')
    parser.add_argument('--image', action='store_true', help="load as image")
    
    args = parser.parse_args()

    gsam = SAM(batch_size=args.batch_size, 
                mask_names=list(set(args.masks)),
                prompts=list(set(args.prompts)))
    
    # filepath = "DatasetB-1/video/001-nm-04-144.avi"
    # jsonpath = "casiab/001-nm-04-144.json"
    # savedir = "outputs/casiab" 
    # if args.image:
    #     gsam.extract_image_masks(args.filepath, jsonpath=args.jsonpath, savedir=args.savedir)
    # else:
        # gsam.extract_video_masks(args.filepath, jsonpath=args.jsonpath, savedir=args.savedir)

    folderpath = "/home/c3-0/datasets/LTCC/LTCC_ReID/query/"
    json_root = "/home/prudvik/sampacs/outputs/LTCC/jsons/query"
    sil_root = "/home/c3-0/datasets/ID-Dataset/ltcc/query/"
    for img_path in tqdm(os.listdir(folderpath)):
        sil_path = os.path.join(sil_root, img_path.replace('.png', '_sil.png'))
        json_path = os.path.join(json_root, img_path.replace('.png', '.json'))
        img_path = os.path.join(folderpath, img_path)
        
        if not os.path.exists(json_path): continue
        if not os.path.exists(sil_path): continue
        gsam.extract_image_masks(img_path, jsonpath=json_path, savedir=args.savedir, prompt_sil=sil_path)
    

# python SAM.py --filepath "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/001-bg-01-000.avi" \
#               --jsonpath "outputs/jsons/001-bg-01-000.json" \
#               --savedir "outputs/silhouettes" \
#               --masks "person" \
#               --masks "shirt" \
#               --masks "pant" \
#               --prompts "bbox"
# python SAM.py --filepath "/home/c3-0/datasets/LTCC/LTCC_ReID/train/094_1_c9_015923.png" --savedir "outputs/silhouettes3" --image \
#                 --masks "person" \
#                 --prompts "bbox" \
#                 --jsonpath "/home/prudvik/sampacs/outputs/jsons/094_1_c9_015923.json"

# python SAM.py --savedir "outputs/LTCC/silhouettes/query" --image --masks "person" --prompts "bbox"
