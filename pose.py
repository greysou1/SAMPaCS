import json, os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

def get_torso_bounding_box(landmarks, width, height, scale_factor_width=1.5, scale_factor_height=1.3):
    # Get the coordinates of the torso landmarks
    torso_landmarks = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                       landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]]

    # Calculate the bounding box coordinates for the torso region
    min_x = min(torso_landmarks, key=lambda landmark: landmark['x'])['x']
    max_x = max(torso_landmarks, key=lambda landmark: landmark['x'])['x']
    min_y = min(torso_landmarks, key=lambda landmark: landmark['y'])['y']
    max_y = max(torso_landmarks, key=lambda landmark: landmark['y'])['y']

    # Calculate the center, width, and height of the bounding box
    center_x = int((min_x + max_x) * width / 2)
    center_y = int((min_y + max_y) * height / 2)
    box_width = int((max_x - min_x) * width)
    box_height = int((max_y - min_y) * height)

    # Scale the width and height of the bounding box separately
    scaled_box_width = int(box_width * scale_factor_width)
    scaled_box_height = int(box_height * scale_factor_height)

    # Calculate the scaled bounding box coordinates
    scaled_min_x = int(center_x - scaled_box_width / 2)
    scaled_max_x = int(center_x + scaled_box_width / 2)
    scaled_min_y = int(center_y - scaled_box_height / 2)
    scaled_max_y = int(center_y + scaled_box_height / 2)

    # Return the scaled bounding box coordinates as tuples
    return (scaled_min_x, scaled_min_y), (scaled_max_x, scaled_max_y)

def get_prompt_point(landmarks, width, height):
    # Get the x and y coordinates of the shoulder and hip keypoints
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate the center point
    center_x = (left_shoulder["x"] + right_shoulder["x"] + left_hip["x"] + right_hip["x"]) / 4
    center_y = (left_shoulder["y"] + right_shoulder["y"] + left_hip["y"] + right_hip["y"]) / 4

    # Convert the relative coordinates of prompt points to pixel coordinates
    center_x = center_x * width
    center_y = center_y * height

    return (center_x, center_y)
    # # Calculate the new points
    # point1_x = center_x - distance_to_move
    # point1_y = center_y
    # point2_x = center_x + distance_to_move
    # point2_y = center_y

    # return ((point1_x, point1_y), (point2_x, point2_y))

def get_pants_bounding_box(landmarks, width, height, scale_factor_width=2, scale_factor_height=1.3):
    # Get the coordinates of the bottom landmarks
    bottom_landmarks = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value],
                        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]]

    # Calculate the bounding box coordinates for the pants region
    min_x = min(bottom_landmarks, key=lambda landmark: landmark['x'])['x']
    max_x = max(bottom_landmarks, key=lambda landmark: landmark['x'])['x']
    min_y = min(bottom_landmarks, key=lambda landmark: landmark['y'])['y']
    max_y = max(bottom_landmarks, key=lambda landmark: landmark['y'])['y']

    # Calculate the center, width, and height of the bounding box
    center_x = int((min_x + max_x) * width / 2)
    center_y = int((min_y + max_y) * height / 2)
    box_width = int((max_x - min_x) * width)
    box_height = int((max_y - min_y) * height)

    # Scale the width and height of the bounding box separately
    scaled_box_width = int(box_width * scale_factor_width)
    scaled_box_height = int(box_height * scale_factor_height)

    # Calculate the scaled bounding box coordinates
    scaled_min_x = int(center_x - scaled_box_width / 2)
    scaled_max_x = int(center_x + scaled_box_width / 2)
    scaled_min_y = int(center_y - scaled_box_height / 2)
    scaled_max_y = int(center_y + scaled_box_height / 2)

    # Return the scaled bounding box coordinates as tuples
    return (scaled_min_x, scaled_min_y), (scaled_max_x, scaled_max_y)
'''
def get_full_body_bounding_box(landmarks, width, height, scale_factor_width=1, scale_factor_height=1):
    # Extract the x and y coordinates of all the landmarks
    landmark_coords = [landmark['x'] for landmark in landmarks] + [landmark['y'] for landmark in landmarks]

    # Calculate the bounding box coordinates for the full body region
    min_x = min(landmark_coords)
    max_x = max(landmark_coords)
    min_y = min(landmark_coords)
    max_y = max(landmark_coords)

    # Calculate the center, width, and height of the bounding box
    center_x = int((min_x + max_x) * width / 2)
    center_y = int((min_y + max_y) * height / 2)
    box_width = int((max_x - min_x) * width)
    box_height = int((max_y - min_y) * height)

    # Swap the x and y coordinates
    center_y, center_x = center_x, center_y
    box_height, box_width = box_width, box_height

    # Scale the width and height of the bounding box separately
    scaled_box_width = int(box_width * scale_factor_width)
    scaled_box_height = int(box_height * scale_factor_height)

    # Calculate the scaled bounding box coordinates
    scaled_min_x = int(center_x - scaled_box_width / 2)
    scaled_max_x = int(center_x + scaled_box_width / 2)
    scaled_min_y = int(center_y - scaled_box_height / 2)
    scaled_max_y = int(center_y + scaled_box_height / 2)

    # Return the scaled bounding box coordinates as tuples
    return (scaled_min_x, scaled_min_y), (scaled_max_x, scaled_max_y)
'''
'''
def get_full_body_bounding_box(landmarks, width, height, scale_factor_width=1.5, scale_factor_height=4):
    # Get the center of the person using the hips
    hips = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]['x']
    center_x = int(sum(hips) * width / 2)
    hips_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]['y']
    center_y = int(sum(hips_y) * height / 2)

    # Calculate the width of the person as the maximum distance between shoulder points and wrist points
    shoulder_points = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]['x']]
    wrist_points = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]['x']]
    distances = [abs(shoulder - wrist) for shoulder in shoulder_points for wrist in wrist_points]
    box_width = int(max(distances) * width)

    # Calculate the height of the person
    shoulder_points_y = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]['y']]
    wrist_points_y = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]['y']]
    hip_points_y = hips_y
    distances_y = [max(shoulder_points_y) - min(wrist_points_y), max(hip_points_y) - min(shoulder_points_y)]
    box_height = int(max(distances_y) * height)

    # Scale the width and height of the bounding box separately
    scaled_box_width = int(box_width * scale_factor_width)
    scaled_box_height = int(box_height * scale_factor_height)

    # Calculate the scaled bounding box coordinates
    scaled_min_x = int(center_x - scaled_box_width / 2)
    scaled_max_x = int(center_x + scaled_box_width / 2)
    scaled_min_y = int(center_y - scaled_box_height / 2)
    scaled_max_y = int(center_y + scaled_box_height / 2)

    # Return the scaled bounding box coordinates as tuples
    return (scaled_min_x, scaled_min_y), (scaled_max_x, scaled_max_y)
'''

def get_full_body_bounding_box(landmarks, width, height, scale_factor_width=1.5, scale_factor_height=1.4):
    # Get the center of the person using the hips
    hips = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]['x']
    center_x = int(sum(hips) * width / 2)
    hips_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]['y']
    center_y = int(sum(hips_y) * height / 2)

    # Calculate the vertical distance between the feet and the top of the head
    foot_points_y = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]['y']]
    face_points_y = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value]['y'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value]['y']]
    distances_y = [min(foot_points_y) - max(face_points_y), max(foot_points_y) - min(face_points_y)]
    box_height = int(max(distances_y) * height)

    # Calculate the width of the person as the maximum distance between shoulder points and wrist points
    shoulder_points = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]['x']]
    wrist_points = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]['x'], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]['x']]
    distances = [abs(shoulder - wrist) for shoulder in shoulder_points for wrist in wrist_points]
    box_width = int(max(distances) * width)

    # Scale the width and height of the bounding box separately
    scaled_box_width = int(box_width * scale_factor_width)
    scaled_box_height = int(box_height * scale_factor_height)

    # Calculate the scaled bounding box coordinates
    scaled_min_x = int(center_x - scaled_box_width / 2)
    scaled_max_x = int(center_x + scaled_box_width / 2)
    scaled_min_y = int(center_y - scaled_box_height / 2)
    scaled_max_y = int(center_y + scaled_box_height / 2)

    # Return the scaled bounding box coordinates as tuples
    return (scaled_min_x, scaled_min_y), (scaled_max_x, scaled_max_y)


def process_video(video_path, output_path, confidence_threshold=0.8):
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
            # print(results.pose_landmarks)
            # quit()
            landmarks = []
            confidences = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z if landmark.HasField('z') else None
                })
                confidences.append(landmark.visibility)

            mean_confidence = np.mean(confidences)

            if mean_confidence >= confidence_threshold:
                person_bb = get_full_body_bounding_box(landmarks, width, height)
                torso_bb = get_torso_bounding_box(landmarks, width, height, scale_factor_width=1.8, scale_factor_height=1.5)
                pants_bb = get_pants_bounding_box(landmarks, width, height)
                prompt_point = get_prompt_point(landmarks, width, height)

                pose_data[frame_count] = {
                    'landmarks': landmarks,
                    'confidences': confidences,
                    'torso_bb': torso_bb,
                    'pants_bb': pants_bb,
                    'person_bb': person_bb,
                    'prompt_point': prompt_point
                }

    return pose_data, width, height, fps

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

filenames = ['084-bg-01-000']

# ## FVG
# filename = "002_01"
# video_path = f"/home/c3-0/datasets/FVG_RGB_vid/session1/{filename}.mp4"
# output_path = f"videooutputs/{filename}_output.mp4"

## CASIA-B 
# filename = "001-bg-01-000"
# video_path = f"/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/{filename}.avi"
# output_path = f"videooutputs/{filename}_output.mp4"

## NTU
# video_path = "/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb/S032C003P008R002A097_rgb.avi"
# output_path = "videooutputs/ntu.mp4"

for filename in filenames:
    sub_id = int(filename.split("-")[0])
    if sub_id < 62:
        video_path= f"/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/{filename}.avi"
    else:
        video_path= f"/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-2/video/{filename}.avi"
    
    output_path = f"videooutputs/{filename}_output.mp4"
    json_output_path = f"jsonoutputs/casiab_debug/{filename}.json"

    pose_data, width, height, fps = process_video(video_path, output_path)

    # Create a dictionary for the video information
    video_info = {
        'width': width,
        'height': height,
        'fps': fps,
        'pose_data': pose_data
    }

    # Save the video information to a JSON file
    with open(json_output_path, 'w') as json_file:
        json.dump(video_info, json_file)

    print(f'Pose data saved to {json_output_path}')
