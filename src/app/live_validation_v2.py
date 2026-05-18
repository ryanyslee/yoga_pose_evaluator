import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import json

# Import your custom math functions
from validate_poses import procrustes_error

# --- Configuration ---
CAMERA_INDEX = 0
STABLE_SECONDS = 5
MOTION_THRESHOLD = 25
MOTION_PERCENT = 2.0
GT_DIR = "GT_Data" 
MODEL_PATH = "pose_landmarker_heavy.task"

# Define the 12 core structural joints
CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Define Angle Definitions for Biomechanical Analysis
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

# --- Helper Functions ---

def detect_motion(frame1, frame2):
    """Calculates pixel difference to detect movement."""
    gray1 = cv2.GaussianBlur(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    gray2 = cv2.GaussianBlur(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    pct = 100 * np.count_nonzero(thresh) / thresh.size
    return pct > MOTION_PERCENT

def compute_angle(a, b, c):
    """Calculates the 3D angle between three points (b is the vertex)."""
    ba = a - b
    bc = c - b
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-8:
        return 0.0
    cosine_angle = np.clip(np.dot(ba, bc) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

def generate_opposite_ground_truth(gt_matrix):
    """Mirrors a 12x3 ground truth matrix to the opposite physical side."""
    flipped_gt = np.copy(gt_matrix)
    
    # 1. Mirror the geometry along the X-axis
    flipped_gt[:, 0] *= -1.0 
    
    # 2. Swap the Left and Right joint pairs (they are adjacent in CORE_JOINTS)
    for i in range(0, len(flipped_gt), 2):
        temp = np.copy(flipped_gt[i])
        flipped_gt[i] = flipped_gt[i + 1]
        flipped_gt[i + 1] = temp
        
    return flipped_gt

def calculate_score(error):
    """Maps the Jacobi SVD error to a 1-5 scale."""
    if error < 0.15: return 5
    elif error < 0.25: return 4
    elif error < 0.35: return 3
    elif error < 0.45: return 2
    else: return 1
    
def generate_coach_feedback(user_angles, gt_angles):
    """Analyzes angles to provide actionable feedback on the worst performing joint."""
    max_diff = 0
    worst_joint = None
    direction_diff = 0
    
    for joint, target_angle in gt_angles.items():
        if target_angle is None or joint not in user_angles:
            continue
            
        user_angle = user_angles[joint]
        diff = target_angle - user_angle
        abs_diff = abs(diff)
        
        # Find the joint furthest from the target
        if abs_diff > max_diff:
            max_diff = abs_diff
            worst_joint = joint
            direction_diff = diff

    # If there are no joints to correct or the error is negligible
    if (worst_joint is None or max_diff < 5.0) and len(user_angles) > 10:
        return "Excellent form!"
    
    # Determine the Adverb based on magnitude
    if max_diff < 30.0:
        adverb = "slightly"
    elif max_diff < 45.0:
        adverb = "moderately"
    else:
        adverb = "greatly"
        
    # Determine the Action
    # If target is 180 and user is 90 (diff is +90) -> User needs to Extend.
    # If target is 90 and user is 180 (diff is -90) -> User needs to Bend.
    action = "Extend" if direction_diff > 0 else "Bend"
    
    return f"Tip: {action} your {worst_joint.lower()} {adverb}"

def draw_core_skeleton(image, results):
    """Draws only the 12 core joints using the new Tasks API structure."""
    if not results or not results.pose_landmarks:
        return
        
    h, w = image.shape[:2]
    lms = results.pose_landmarks[0] 
    
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
        (11, 23), (12, 24), (23, 24), 
        (23, 25), (25, 27), (24, 26), (26, 28)
    ]
    
    for start_idx, end_idx in CONNECTIONS:
        start_lm = lms[start_idx]
        end_lm = lms[end_idx]
        
        if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
            pt1 = (int(start_lm.x * w), int(start_lm.y * h))
            pt2 = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
            
    for idx in CORE_JOINTS:
        lm = lms[idx]
        if lm.visibility > 0.5:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, pt, 6, (0, 0, 255), -1)

def extract_and_evaluate_frame(frame, gt_matrix_full, visibility_threshold=0.50):
    """Extracts joints, calculates angles, and dynamically slices matrices."""
    # RAW UNFLIPPED frame passed here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    results = landmarker.detect(mp_image)
    
    if not results.pose_landmarks:
        return None, None, None, None
        
    lms = results.pose_landmarks[0]
    user_angles = {}
    
    is_2d = (gt_matrix_full.shape[1] == 2)
    
    for joint_name, (a_idx, b_idx, c_idx) in ANGLE_DEFS.items():
        if (lms[a_idx].visibility > visibility_threshold and 
            lms[b_idx].visibility > visibility_threshold and 
            lms[c_idx].visibility > visibility_threshold):
            
            # Extract only 2D coords if it's a 2D pose
            if is_2d:
                a = np.array([lms[a_idx].x, lms[a_idx].y])
                b = np.array([lms[b_idx].x, lms[b_idx].y])
                c = np.array([lms[c_idx].x, lms[c_idx].y])
            else:
                a = np.array([lms[a_idx].x, lms[a_idx].y, lms[a_idx].z])
                b = np.array([lms[b_idx].x, lms[b_idx].y, lms[b_idx].z])
                c = np.array([lms[c_idx].x, lms[c_idx].y, lms[c_idx].z])
                
            user_angles[joint_name] = compute_angle(a, b, c)


    valid_indices = []
    landmarks_extracted = []
    
    for i, mp_idx in enumerate(CORE_JOINTS):
        # Check if the Ground Truth matrix has a [0, 0, 0]
        gt_point = gt_matrix_full[i]
        
        # Dimension-safe check for missing joint bombs
        if is_2d:
            is_gt_valid = not (gt_point[0] == 0.0 and gt_point[1] == 0.0)
        else:
            is_gt_valid = not (gt_point[0] == 0.0 and gt_point[1] == 0.0 and gt_point[2] == 0.0)
        
        if lms[mp_idx].visibility > visibility_threshold and is_gt_valid:
            valid_indices.append(i) 
            if is_2d:
                landmarks_extracted.append([lms[mp_idx].x, lms[mp_idx].y])
            else:
                landmarks_extracted.append([lms[mp_idx].x, lms[mp_idx].y, lms[mp_idx].z])
        
    if len(valid_indices) < 4: 
        return None, None, None, None
        
    A_extracted = np.array(landmarks_extracted)
    B_sliced = gt_matrix_full[valid_indices, :] 
    
    A_centered = A_extracted - np.mean(A_extracted, axis=0)
    A_scale = np.linalg.norm(A_centered, 'fro')
    A_norm = A_centered / A_scale if A_scale != 0 else A_centered
    
    B_centered = B_sliced - np.mean(B_sliced, axis=0)
    B_scale = np.linalg.norm(B_centered, 'fro')
    B_norm = B_centered / B_scale if B_scale != 0 else B_centered
    
    return A_norm, B_norm, user_angles, results

# --- Main Execution ---

def main():
    pose_configs = {
        "1": {"name": "balancing_table", "gt_file": "balancing_table_gt.npy", "ref_img": "balancing_table_ref.jpg"},
        "2": {"name": "cobra", "gt_file": "cobra_gt.npy", "ref_img": "cobra_ref.jpg"},
        "3": {"name": "downward_facing_dog", "gt_file": "downward_facing_dog_gt.npy", "ref_img": "downward_facing_dog_ref.jpg"},
        "4": {"name": "one_legged_king_pigeon", "gt_file": "one_legged_king_pigeon_gt.npy", "ref_img": "one_legged_king_pigeon_ref.jpg"},
        "5": {"name": "tree", "gt_file": "tree_gt.npy", "ref_img": "tree_ref.jpg"},
        "6": {"name": "triangle", "gt_file": "triangle_gt.npy", "ref_img": "triangle_ref.jpg"}
    }
    
    quit_application = False
    
    # OUTER LOOP: Main Menu Navigation
    while not quit_application:
        print("\n" + "="*60)
        print("🧘 YOGA POSE ALIGNMENT SYSTEM")
        print("="*60)
        
        # 1. Pose Selection
        for key, val in pose_configs.items():
            print(f"[{key}] {val['name'].replace('_',' ').title()}")
            
        print("\n[Q] Quit Application")
        choice = input("\nEnter the number of the pose you want to perform: ").strip().lower()
            
        if choice == 'q':
            print("Exiting application. Namaste!")
            break
            
        if choice not in pose_configs:
            print("❌ Invalid choice. Exiting.")
            return
            
        target_pose = pose_configs[choice]["name"]
        gt_file = pose_configs[choice]["gt_file"]
        ref_img_name = pose_configs[choice]["ref_img"]
        
        # 2. Side Selection
        print(f"\nPose selected: {target_pose.replace('-', ' ').title()}")
        print("Which side would you like to practice?")
        print("[1] Orientation 1 (Mirror default photo)")
        print("[2] Orientation 2 (Mirror opposite side)")
        side_choice = input("Enter 1 or 2: ").strip()
        mirror_default = (side_choice == "1")
        
        gt_file_path = os.path.join(GT_DIR, target_pose, gt_file)
        gt_json_path = os.path.join(GT_DIR, target_pose, gt_file.replace('.npy', '_angles.json'))
        ref_img_path = os.path.join(GT_DIR, target_pose, ref_img_name)
        
        try:
            gt_matrix = np.load(gt_file_path) 
            if mirror_default:
                gt_matrix = generate_opposite_ground_truth(gt_matrix)
                print(f"\n🔄 Loaded and flipped Ground Truth matrix to match your mirror movement.")
            else:
                print(f"\n✅ Loaded standard Ground Truth matrix to match your mirror movement.")
        except FileNotFoundError:
            print(f"\n❌ Error: Could not find ground truth file '{gt_file_path}'.")
            continue

        # Load and potentially mirror Angles
        try:
            with open(gt_json_path, 'r') as f:
                raw_angles = json.load(f)
                
            if mirror_default:
                # Swap "Left" and "Right" keys to match the mirrored user
                gt_angles = {}
                for k, v in raw_angles.items():
                    if "Left" in k:
                        gt_angles[k.replace("Left", "Right")] = v
                    elif "Right" in k:
                        gt_angles[k.replace("Right", "Left")] = v
                    else:
                        gt_angles[k] = v
            else:
                gt_angles = raw_angles
        except FileNotFoundError:
            print(f"\n⚠️ Warning: Could not find ground truth angles '{gt_json_path}'. Coaching disabled.")
            gt_angles = {}

        # Load Reference Image
        ref_image = cv2.imread(ref_img_path)
        if ref_image is None:
            print(f"⚠️ Warning: Reference image '{ref_img_name}' not found. Using placeholder.")
            ref_image = np.zeros((480, 360, 3), dtype=np.uint8)
            cv2.putText(ref_image, "Reference Image", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        elif not mirror_default:
            ref_image = cv2.flip(ref_image, 1)

        # Start Webcam
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened(): 
            print("Error: Could not open webcam.")
            continue

        print(f"\nStarting webcam... Get into position and match the screen!")
        
        prev_frame = None
        stable_since = None
        captured = False
        result_frame = None

        # INNER LOOP: Webcam Feed
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            h, w = frame.shape[:2]
            now = time.time()
            
            # --- KEYBOARD CONTROLS ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                quit_application = True # Break both loops
                break
            elif key == ord('m'):
                print("\nReturning to Main Menu...")
                break # Break inner loop, return to menu

            if result_frame is not None:
                # --- EVALUATION STATE ---
                display = result_frame.copy()
                cv2.rectangle(display, (0, h-40), (w, h), (0,0,0), -1)
                cv2.putText(display, "SPACE: Retry | M: Menu | Q: Quit", (10, h-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if key == ord(' '):
                    result_frame = None
                    captured = False
                    stable_since = None
                    continue
            else:
                # --- LIVE TRACKING STATE ---
                if prev_frame is not None:
                    is_moving = detect_motion(prev_frame, frame)
                    if is_moving: stable_since = None
                    elif stable_since is None: stable_since = now
                prev_frame = frame.copy()

                display = cv2.flip(frame.copy(), 1)
                cv2.rectangle(display, (0, h-40), (w, h), (0,0,0), -1)

                if stable_since is not None:
                    elapsed = now - stable_since
                    remaining = max(0, STABLE_SECONDS - elapsed)
                    bar_w = min(int((elapsed/STABLE_SECONDS)*(w-40)), w-40)
                    
                    cv2.rectangle(display, (20, h-30), (w-20, h-10), (60, 60, 60), -1)
                    cv2.rectangle(display, (20, h-30), (20+bar_w, h-10), (0, 220, 0), -1)
                    cv2.putText(display, f"Hold still! Analyzing in {remaining:.1f}s... (M for Menu)", 
                                (25, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if elapsed >= STABLE_SECONDS and not captured:
                        captured = True
                        print(f"\n📸 Snapshot taken! Calculating errors...")
                        
                        A_norm, B_norm, user_angles, mp_results = extract_and_evaluate_frame(frame, gt_matrix)
                        
                        if A_norm is not None:
                            np_err, jacobi_err = procrustes_error(A_norm, B_norm)
                            score = calculate_score(jacobi_err)
                            
                            print("\n" + "-" * 40)
                            print(f"POSE: {target_pose.replace('_', ' ').title()}")
                            print(f"JACOBI SVD STRUCTURAL ERROR: {jacobi_err:.4f} -> SCORE: {score}/5")
                            print("-" * 40)
                            print("BIOMECHANICAL ANGLES:")
                            for joint_name, user_val in user_angles.items():
                                print(f"  {joint_name}: {user_val:.1f}°")
                            print("-" * 40)
                            
                            draw_core_skeleton(frame, mp_results)
                            result_frame = cv2.flip(frame, 1)
                            
                            # Select Color based on Score
                            if score >= 4:
                                color = (0, 255, 0) # Green
                            elif score == 3:
                                color = (0, 255, 255) # Yellow
                            else:
                                color = (0, 0, 255) # Red
                                
                            # Create a larger UI Box for Score + Coaching
                            cv2.rectangle(result_frame, (5, 5), (600, 100), (0,0,0), -1)
                            
                            cv2.putText(result_frame, f"Score: {score}/5", (15, 45), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                            
                            if score <=4 and gt_angles:
                                tip = generate_coach_feedback(user_angles, gt_angles)
                                cv2.putText(result_frame, tip, (15, 85),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            elif score == 5:
                                cv2.putText(result_frame, "Perfect form! Great job!", (15, 85),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                        else:
                            print("❌ No body detected or too many hidden joints. Try again.")
                            captured = False
                            stable_since = None
                else:
                    cv2.putText(display, "Match the screen | M: Menu | Q: Quit", (10, h-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            # --- THE SCALING ENGINE ---
            SCALE_FACTOR = 2.0
            REF_RATIO = 0.55
            
            display_scaled = cv2.resize(display, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            live_h, live_w = display_scaled.shape[:2]
            
            # Scale reference image to match the height of the webcam feed
            ref_h_orig, ref_w_orig = ref_image.shape[:2]
            aspect_ratio = ref_w_orig / ref_h_orig
            
            target_ref_h = int(live_h * REF_RATIO)
            target_ref_w = int(target_ref_h * aspect_ratio)
            ref_scaled = cv2.resize(ref_image, (target_ref_w, target_ref_h))
            
            ref_canvas = np.zeros((live_h, target_ref_w, 3), dtype=np.uint8)
            
            y_offset = (live_h - target_ref_h) // 2
            ref_canvas[y_offset:y_offset+target_ref_h, 0:target_ref_w] = ref_scaled
            
            combined_display = cv2.hconcat([display_scaled, ref_canvas])
            
            cv2.namedWindow("Yoga Alignment Tracker", cv2.WINDOW_NORMAL)
            cv2.imshow("Yoga Alignment Tracker", combined_display)

        # Clean up the camera resources before returning to the menu or exiting
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    main()