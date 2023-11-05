import gc, os
import cv2
import numpy as np
from tqdm import tqdm

import torch
from segment_anything import sam_model_registry
import moviepy.video.io.ImageSequenceClip

class SAM:
    def __init__(self, 
                 sam_checkpoint="sam_vit_h_4b8939.pth",
                 device='cuda',
                 batch_size=1,
                 mask_names=["person", "shirt", "pant"]):

        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self.batch_size = batch_size
        self.mask_names = mask_names
        
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
            new_data[int(key)]['person_bb'] = [item for sublist in item['person_bb'] for item in sublist]
            new_data[int(key)]['torso_bb'] = [item for sublist in item['torso_bb'] for item in sublist]
            new_data[int(key)]['pants_bb'] = [item for sublist in item['pants_bb'] for item in sublist]
            
            # the center point of torso to use as prompt point
            new_data[int(key)]['prompt_point'] = item['prompt_point']
            # (x, y) values of 22 landmarks that can be used as prompt points
            new_data[int(key)]['landmarks'] = [[cd['x'] * w, cd['y'] * h] for sublist in item['landmarks'] for cd in sublist] # cd: coords_dict

        return new_data
    
    def extract_video_masks(self, videopath, 
                            jsonpath=None, 
                            savedir=None, scaling_factor=1,
                            prompts=['bbox', 'pose']):
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

        videoname = videopath.split('/')[-1].split('.')[0]
        # sub_id = videoname.split('-')[0] # 023
        # view_angle = videoname.split('-')[-1] # 090
        # cond = videoname.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

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
        bbox_savepath = os.path.join(savedir, "clothing-jsons")
        if not os.path.exists(bbox_savepath): os.makedirs(bbox_savepath, exist_ok=True)
        bbox_savepath = os.path.join(bbox_savepath, f"{videoname}.json")

        # b. save the masks:  clothing_masks
        mask_save_fol_paths = {}
        for item in self.mask_names:
            savepath = os.path.join(savedir, f"silhouettes-{item}")
            if not os.path.exists(savepath): os.makedirs(savepath, exist_ok=True)
            mask_save_fol_paths[item] = savepath

        # 3. batch inference
        for i in tqdm(range(0, len(prompt_data), self.batch_size)):
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

                if 'bbox' in prompts:
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

                if 'point' in prompts:
                    prompt_point = [item*scaling_factor for item in prompt_point]
                    input_data['input_point'] =  np.array([prompt_point])
                    input_data['input_label'] =  np.array([1]) # 1 is for foreground, 0 is for background
                elif 'pose' in prompts:
                    prompt_points = [item*scaling_factor for item in pose_points]
                    input_data['input_point'] =  np.array([prompt_points])
                    input_data['input_label'] =  np.array([1])

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
        
        savepaths = [bbox_savepath, mask_save_fol_paths]
        self.save_clothing_data(videoname, clothing_bboxes, clothing_masks, savepaths)
    
    def save_clothing_data(self, videoname, clothing_bboxes, clothing_masks, savepaths):
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
        bbox_savepath, mask_save_paths_vid = savepaths

        # clothing_masks[0].keys() -> "person", "shirt", "pant"
        # mask = list(clothing_masks[1].values())[0]
        # frame_height, frame_width = mask.shape[-2:]

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
                
                if mask_item == "shirt":
                    cv2.imwrite("outputs/debug.png", mask_img)
                # Write the frames to the output video
                image_list[mask_item].append(mask_img)
                
        fps = 15
        
        for mask_name in self.mask_names:
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_list[mask_name], fps=fps)
            clip.write_videofile(os.path.join(mask_save_paths_vid[mask_name], f"{videoname}.mp4"), fps=fps, codec="libx264")
        
        # ------------------------ save bboxes ------------------------
        json_dict = {}
        for index, clothing_bbox in clothing_bboxes.items():
            bboxes = []
            for bbox_item, bbox in clothing_bbox.items():
                bboxes.append({
                    'label': bbox_item,
                    'box': bbox
                })
            json_dict[index] = bboxes

        with open(bbox_savepath, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f)
        # -------------------------------------------------------------

if __name__ == "__main__":    
    gsam = SAM(batch_size=1, 
                mask_names=["person"])
    
    ## FVG
    # filename = "002_01"
    # video_file = f"/home/c3-0/datasets/FVG_RGB_vid/session1/{filename}.mp4"
    # personbboxes_jsonpath = f"/home/c3-0/datasets/FVG_GSAM_sill/session1/json/{filename}.json"
    # clothingbboxes_jsonpath = f"/home/prudvik/id-dataset/pose-detection/jsonoutputs/{filename}_output.json"
    # savedir = "outputs/fvg-debug" 

    ## CASIA-B 
    filenames = ['001-nm-04-144', '082-nm-03-054', '082-nm-05-180', '082-nm-03-090', '082-nm-03-072',
                 '082-nm-03-180', '001-nm-04-162', '082-nm-02-036', '001-nm-04-054', '001-nm-03-018', 
                 '026-nm-05-000', '082-nm-03-126', '001-nm-04-018', '082-nm-02-054', '082-nm-05-018', 
                 '084-bg-01-108', '001-nm-03-000', '069-cl-01-018', '084-bg-01-018', '001-bg-02-126', 
                 '082-nm-02-018', '082-nm-05-000', '001-nm-04-036', '084-bg-01-072', '001-nm-03-036', 
                 '021-nm-06-162', '001-nm-03-162', '082-nm-03-144', '001-nm-03-180', '084-bg-01-000', 
                 '026-nm-05-162', '082-nm-03-000', '082-nm-02-180', '082-nm-03-036', '001-nm-03-054', 
                 '063-nm-01-162', '001-nm-04-000', '001-nm-03-144', '082-nm-02-126', '082-nm-05-162', 
                 '084-bg-01-180', '082-nm-03-108', '084-bg-01-144', '082-nm-05-054', '082-nm-02-144', 
                 '026-nm-05-180', '084-bg-01-162', '022-nm-05-162', '001-nm-04-180', '082-nm-05-144', 
                 '082-nm-03-018', '082-nm-02-000', '082-nm-02-072', '082-nm-02-090', '084-bg-01-054', 
                 '110-cl-02-126', '082-nm-05-036', '082-nm-02-108', '082-nm-03-162', '082-nm-02-162']
    
    filenames = ['001-nm-04-144']

    for filename in filenames:
        print(filename)
        video_file = f"/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/{filename}.avi"
        b = filename.split("-")

        save_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes2"
        clothingbboxes_jsonpath = f"/home/prudvik/id-dataset/pose-detection/jsonoutputs/casiab/{b[0]}-{b[1]}-{b[2]}-{b[3]}.json"
        
        savedir = "outputs/casiab" 

        gsam.extract_video_masks(video_file,
                                 savedir=savedir,
                                 personbboxes_jsonpath=personbboxes_jsonpath,
                                 clothingbboxes_jsonpath=clothingbboxes_jsonpath)
