import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def visualize_core_joints(image_path, save_output=True):
    """Draws the 12 core structural joints and skeleton on the given image."""
    print(f"Processing: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"  -> Error: Could not load image at {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print("  -> Error: No pose detected in the image.")
        return

    # Get image dimensions for coordinate conversion
    h, w, c = image.shape

    # Define the 12 core structural joints
    CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
    # Define the skeletal connections between these specific joints
    CONNECTIONS = [
        (11, 12), # Shoulders
        (11, 13), (13, 15), # Left Arm
        (12, 14), (14, 16), # Right Arm
        (11, 23), (12, 24), # Torso (Shoulders to Hips)
        (23, 24), # Hips
        (23, 25), (25, 27), # Left Leg
        (24, 26), (26, 28)  # Right Leg
    ]

    # 1. Draw the skeletal lines first (so they are behind the dots)
    for start_idx, end_idx in CONNECTIONS:
        start_lm = results.pose_landmarks.landmark[start_idx]
        end_lm = results.pose_landmarks.landmark[end_idx]
        
        # Convert normalized coordinates to pixel coordinates
        start_point = (int(start_lm.x * w), int(start_lm.y * h))
        end_point = (int(end_lm.x * w), int(end_lm.y * h))
        
        cv2.line(image, start_point, end_point, (255, 255, 255), 2) # White lines

    # 2. Draw the joints and labels
    for idx in CORE_JOINTS:
        landmark = results.pose_landmarks.landmark[idx]
        cx, cy = int(landmark.x * w), int(landmark.y * h)

        # Draw a bright green circle at the joint
        cv2.circle(image, (cx, cy), radius=6, color=(0, 255, 0), thickness=-1)

        # Draw the MediaPipe index number in red next to the joint
        cv2.putText(image, str(idx), (cx + 10, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 3. Display and Save the image
    # Note: cv2.imshow can sometimes hang on macOS terminals. 
    # Saving a copy ensures you can always open and view it.
    if save_output:
        output_filename = "visualized_" + image_path.split('/')[-1]
        cv2.imwrite(output_filename, image)
        print(f"  -> Saved visualization to: {output_filename}")

    cv2.imshow('Core Joints Verification', image)
    print("  -> Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test it on one of your problematic images
    # test_image_path = "Test Poses/olkp_pose/olkp_test_2.jpg" 
    # test_image_path = "Test Poses/tree_pose/tree_test_2.jpg" 
    test_image_path = "Yoga Poses/olkp_pose/olkp_pose_3.jpg" 
    
    visualize_core_joints(test_image_path)