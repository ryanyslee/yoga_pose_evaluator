import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import os
import glob
import json

# --- Configuration ---
MODEL_PATH = "pose_landmarker_heavy.task"
BASE_DIR = "raw_images" # Folder containing subfolders of your 5 images
VISIBILITY_THRESHOLD = 0.15

# The 12 core structural joints
CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Angle Definitions
ANGLE_DEFS = {
    "Left Elbow": (11, 13, 15),
    "Right Elbow": (12, 14, 16),
    "Left Shoulder": (13, 11, 23),
    "Right Shoulder": (14, 12, 24),
    "Left Hip": (11, 23, 25),
    "Right Hip": (12, 24, 26),
    "Left Knee": (23, 25, 27),
    "Right Knee": (24, 26, 28)
}

# --- Initialize MediaPipe Tasks API ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

def compute_angle(a, b, c):
    """Calculates the angle between three points. Works dynamically for 2D or 3D vectors."""
    ba = a - b
    bc = c - b
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-8:
        return 0.0
    cosine_angle = np.clip(np.dot(ba, bc) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

def build_ground_truth_for_pose(pose_folder_path, pose_name):
    """Processes 5 images to create an averaged matrix and average angles."""
    
    image_files = glob.glob(os.path.join(pose_folder_path, "*.[jp][pn]*[g]"))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {pose_folder_path}")
        return

    # Check if this pose should be strictly 2D
    is_2d_pose = pose_name in ["tree", "triangle"]
    num_dims = 2 if is_2d_pose else 3
    
    print(f"\nProcessing {len(image_files)} images for {pose_name} (Dimensions: {num_dims}D)...")

    # Accumulators for Smart Averaging
    joint_accumulators = {idx: [] for idx in CORE_JOINTS}
    angle_accumulators = {name: [] for name in ANGLE_DEFS.keys()}

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = landmarker.detect(mp_image)
        
        if not results.pose_landmarks:
            print(f"  -> Warning: No pose detected in {os.path.basename(img_path)}")
            continue
            
        lms = results.pose_landmarks[0]
        
        # 1. Accumulate Coordinates (Flattening Z if it's a 2D pose)
        for mp_idx in CORE_JOINTS:
            lm = lms[mp_idx]
            if lm.visibility > VISIBILITY_THRESHOLD:
                if is_2d_pose:
                    joint_accumulators[mp_idx].append([lm.x, lm.y])
                else:
                    joint_accumulators[mp_idx].append([lm.x, lm.y, lm.z])
                
        # 2. Accumulate Biomechanical Angles (using 2D or 3D math respectively)
        for joint_name, (a_idx, b_idx, c_idx) in ANGLE_DEFS.items():
            if (lms[a_idx].visibility > VISIBILITY_THRESHOLD and 
                lms[b_idx].visibility > VISIBILITY_THRESHOLD and 
                lms[c_idx].visibility > VISIBILITY_THRESHOLD):
                
                if is_2d_pose:
                    a = np.array([lms[a_idx].x, lms[a_idx].y])
                    b = np.array([lms[b_idx].x, lms[b_idx].y])
                    c = np.array([lms[c_idx].x, lms[c_idx].y])
                else:
                    a = np.array([lms[a_idx].x, lms[a_idx].y, lms[a_idx].z])
                    b = np.array([lms[b_idx].x, lms[b_idx].y, lms[b_idx].z])
                    c = np.array([lms[c_idx].x, lms[c_idx].y, lms[c_idx].z])
                    
                angle = compute_angle(a, b, c)
                angle_accumulators[joint_name].append(angle)

    # --- CALCULATE AVERAGES ---
    
    # Average the Matrix (Dynamically sized to 12x2 or 12x3)
    gt_matrix = np.zeros((12, num_dims))
    for i, mp_idx in enumerate(CORE_JOINTS):
        valid_points = joint_accumulators[mp_idx]
        if len(valid_points) > 0:
            gt_matrix[i] = np.mean(valid_points, axis=0)
        else:
            print(f"  -> CRITICAL WARNING: Joint {mp_idx} was hidden in ALL training images.")
            # Dynamic fallback bomb
            gt_matrix[i] = [0.0, 0.0] if is_2d_pose else [0.0, 0.0, 0.0]

    # Average the Angles
    gt_angles = {}
    for joint_name in ANGLE_DEFS.keys():
        valid_angles = angle_accumulators[joint_name]
        if len(valid_angles) > 0:
            gt_angles[joint_name] = round(float(np.mean(valid_angles)), 2)
        else:
            gt_angles[joint_name] = None # Hidden in all images

    # --- SAVE TO DISK ---
    
    # 1. Save Matrix (.npy)
    npy_filename = f"{pose_name}_gt.npy"
    np.save(npy_filename, gt_matrix)
    print(f"✅ Saved structural matrix to {npy_filename}")
    
    # 2. Save Angles (.json)
    json_filename = f"{pose_name}_gt_angles.json"
    with open(json_filename, 'w') as f:
        json.dump(gt_angles, f, indent=4)
    print(f"✅ Saved biomechanical angles to {json_filename}")

if __name__ == "__main__":
    
    # Map your folder names to the desired output prefixes
    pose_folders = {
        "balancing_table": "balancing_table",
        "cobra": "cobra",
        "downward_facing_dog": "downward_facing_dog",
        "one_legged_king_pigeon": "one_legged_king_pigeon",
        "tree": "tree",
        "triangle": "triangle"
    }
    
    for folder_name, pose_prefix in pose_folders.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if os.path.exists(folder_path):
            build_ground_truth_for_pose(folder_path, pose_prefix)
        else:
            print(f"⚠️ Could not find directory: {folder_path}")