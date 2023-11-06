import json, os
import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

def save_json(data, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)

    if args.verbose:
        print(f'Pose data saved to {json_path}')

def get_torso_bounding_box(landmarks, width, height, scale_width=1.5, scale_height=1.3):
    # get the coordinates of the torso landmarks
    torso_landmarks = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                       landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]]

    # calculate the bounding box coordinates for the torso region
    min_x = min(torso_landmarks, key=lambda landmark: landmark['x'])['x']
    max_x = max(torso_landmarks, key=lambda landmark: landmark['x'])['x']
    min_y = min(torso_landmarks, key=lambda landmark: landmark['y'])['y']
    max_y = max(torso_landmarks, key=lambda landmark: landmark['y'])['y']

    # calculate the center, width, and height of the bounding box
    center_x = int((min_x + max_x) * width / 2)
    center_y = int((min_y + max_y) * height / 2)
    box_width = int((max_x - min_x) * width)
    box_height = int((max_y - min_y) * height)

    scaled_box_width = int(box_width * scale_width)
    scaled_box_height = int(box_height * scale_height)

    # calculate the scaled bounding box coordinates
    scaled_min_x = int(center_x - scaled_box_width / 2)
    scaled_max_x = int(center_x + scaled_box_width / 2)
    scaled_min_y = int(center_y - scaled_box_height / 2)
    scaled_max_y = int(center_y + scaled_box_height / 2)

    return (scaled_min_x, scaled_min_y), (scaled_max_x, scaled_max_y)

def get_prompt_point(landmarks, width, height):
    # get the x and y coordinates of the shoulder and hip keypoints
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    # calculate the center point
    center_x = (left_shoulder["x"] + right_shoulder["x"] + left_hip["x"] + right_hip["x"]) / 4
    center_y = (left_shoulder["y"] + right_shoulder["y"] + left_hip["y"] + right_hip["y"]) / 4

    center_x = center_x * width
    center_y = center_y * height

    return (center_x, center_y)

def get_pants_bounding_box(landmarks, width, height, scale_width=2, scale_height=1.3):
    # get the coordinates of the bottom landmarks
    bottom_landmarks = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value],
                        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]]

    # calculate the bounding box coordinates for the pants region
    min_x = min(bottom_landmarks, key=lambda landmark: landmark['x'])['x']
    max_x = max(bottom_landmarks, key=lambda landmark: landmark['x'])['x']
    min_y = min(bottom_landmarks, key=lambda landmark: landmark['y'])['y']
    max_y = max(bottom_landmarks, key=lambda landmark: landmark['y'])['y']

    # calculate the center, width, and height of the bounding box
    center_x = int((min_x + max_x) * width / 2)
    center_y = int((min_y + max_y) * height / 2)
    box_width = int((max_x - min_x) * width)
    box_height = int((max_y - min_y) * height)

    scaled_box_width = int(box_width * scale_width)
    scaled_box_height = int(box_height * scale_height)

    # calculate the scaled bounding box coordinates
    scaled_min_x = int(center_x - scaled_box_width / 2)
    scaled_max_x = int(center_x + scaled_box_width / 2)
    scaled_min_y = int(center_y - scaled_box_height / 2)
    scaled_max_y = int(center_y + scaled_box_height / 2)

    return (scaled_min_x, scaled_min_y), (scaled_max_x, scaled_max_y)

def get_full_body_bounding_box(landmarks, width, height, scale_width=1.5, scale_height=1.4):
    # get the center of the person using the hips
    hips = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]['x']
    center_x = int(sum(hips) * width / 2)
    hips_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]['y']
    center_y = int(sum(hips_y) * height / 2)

    # calculate the distance between the feet and the top of the head
    foot_points_y = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]['y']]
    face_points_y = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value]['y']]
    distances_y = [min(foot_points_y) - max(face_points_y), max(foot_points_y) - min(face_points_y)]
    box_height = int(max(distances_y) * height)

    # calculate the width of the person as the maximum distance between shoulder points and wrist points
    shoulder_points = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]['x']]
    wrist_points = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]['x']]
    distances = [abs(shoulder - wrist) for shoulder in shoulder_points for wrist in wrist_points]
    box_width = int(max(distances) * width)

    scaled_box_width = int(box_width * scale_width)
    scaled_box_height = int(box_height * scale_height)

    # calculate the scaled bounding box coordinates
    scaled_min_x = int(center_x - scaled_box_width / 2)
    scaled_max_x = int(center_x + scaled_box_width / 2)
    scaled_min_y = int(center_y - scaled_box_height / 2)
    scaled_max_y = int(center_y + scaled_box_height / 2)

    return (scaled_min_x, scaled_min_y), (scaled_max_x, scaled_max_y)

def process_video(video_path, json_dir,
                  confidence_threshold=0.8,
                  scale_torso_width=1.8, scale_pants_width=2, scale_person_width=1.5,
                  scale_torso_height=1.5, scale_pants_height=1.3, scale_person_height=1.4):

    mp_pose = mp.solutions.pose.Pose()

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_data = {}

    for frame_count in tqdm(range(num_frames)):
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = mp_pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            landmarks = []
            confidences = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': round(landmark.x, 5),
                    'y': round(landmark.y, 5),
                })
                confidences.append(round(landmark.visibility, 4))

            mean_confidence = np.mean(confidences)

            if mean_confidence >= confidence_threshold:
                prompt_point = get_prompt_point(landmarks, width, height)
                torso_bb = get_torso_bounding_box(landmarks, width, height, scale_width=scale_torso_width, scale_height=scale_torso_height)
                pants_bb = get_pants_bounding_box(landmarks, width, height, scale_width=scale_pants_width, scale_height=scale_pants_height)
                person_bb = get_full_body_bounding_box(landmarks, width, height, scale_width=scale_person_width, scale_height=scale_person_height)
                
                pose_data[frame_count] = {
                    'torso_bb': torso_bb,
                    'pants_bb': pants_bb,
                    'person_bb': person_bb,
                    'landmarks': landmarks,
                    'confidences': confidences,
                    'prompt_point': prompt_point
                }

    video_info = {
        'fps': fps,
        'width': width,
        'height': height,
        'pose_data': pose_data
    }

    json_path = os.path.join(json_dir, Path(video_path).stem+".json")
    save_json(video_info, json_path)
    return video_info

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

    process_video(args.filepath, args.jsonsavedir, confidence_threshold=args.confidence,
                  scale_torso_width=args.scale_torso_width, scale_torso_height=args.scale_torso_height,
                  scale_pants_width=args.scale_pants_width, scale_pants_height=args.scale_pants_height,
                  scale_person_width=args.scale_person_width, scale_person_height=args.scale_person_height)


# python pose.py --filepath "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/084-bg-01-000.avi" --jsonsavedir "outputs/jsons"