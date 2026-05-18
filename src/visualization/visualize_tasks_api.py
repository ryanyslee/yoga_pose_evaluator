import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os

# --- Configuration ---
# MODEL_PATH = "pose_landmarker.task" # Ensure this file is in the same folder
MODEL_PATH = "pose_landmarker_heavy.task"
IMAGE_PATH = os.path.join("Test Poses", "olkp_pose", "olkp_test_1.jpg")       # Change this to the image you want to test!

def visualize_hidden_joints(image_path, model_path):
    # 1. Initialize Tasks API
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1
    )
    
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        # 2. Load Image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image at '{image_path}'")
            return
            
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # 3. Process Image
        print("🧠 Processing image with Tasks API...")
        results = landmarker.detect(mp_image)
        
        if not results.pose_landmarks:
            print("❌ No pose detected in the image.")
            return
            
        lms = results.pose_landmarks[0]
        
        # 4. Visualization Setup
        annotated_image = image.copy()
        h, w = annotated_image.shape[:2]
        
        # Grab the standard 33-joint connection map
        mp_pose = mp.solutions.pose
        connections = mp_pose.POSE_CONNECTIONS
        
        # --- DRAW BONES ---
        for connection in connections:
            start_idx, end_idx = connection
            start_lm = lms[start_idx]
            end_lm = lms[end_idx]
            
            pt1 = (int(start_lm.x * w), int(start_lm.y * h))
            pt2 = (int(end_lm.x * w), int(end_lm.y * h))
            
            # If BOTH joints are highly visible -> Green Bone
            # If AT LEAST ONE joint is hidden/estimated -> Red Bone
            if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                cv2.line(annotated_image, pt1, pt2, (0, 255, 0), 2)
            else:
                cv2.line(annotated_image, pt1, pt2, (0, 0, 255), 2)
                
        # --- DRAW JOINTS ---
        for idx, lm in enumerate(lms):
            pt = (int(lm.x * w), int(lm.y * h))
            
            if lm.visibility > 0.5:
                # Visible joint -> Green circle
                cv2.circle(annotated_image, pt, 5, (0, 255, 0), -1)
            else:
                # Hidden/Estimated joint -> Red circle with visibility score
                cv2.circle(annotated_image, pt, 6, (0, 0, 255), -1)
                cv2.putText(annotated_image, f"v:{lm.visibility:.2f}", (pt[0]+8, pt[1]-8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # 5. Display and Save Result
        output_filename = "tasks_api_estimation_heavy.jpg"
        cv2.imwrite(output_filename, annotated_image)
        print(f"✅ Saved output visualization to '{output_filename}'")
        
        # Show on screen (Press any key to close)
        cv2.imshow("Tasks API - Hidden Joint Estimation", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_hidden_joints(IMAGE_PATH, MODEL_PATH)