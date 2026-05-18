import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def get_normalized_pose_matrices(image_path):
    """Extracts, centers, and scales both 2D and 3D poses from a single image."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, None
        
    # Define the 12 core structural joints
    CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
    landmarks_2d, landmarks_3d = [], []
    for idx in CORE_JOINTS:
        landmark = results.pose_landmarks.landmark[idx]
        landmarks_2d.append([landmark.x, landmark.y])
        landmarks_3d.append([landmark.x, landmark.y, landmark.z])
    
    A_2d, A_3d = np.array(landmarks_2d), np.array(landmarks_3d)
    
    # Normalize 2D
    A_centered_2d = A_2d - np.mean(A_2d, axis=0)
    scale_2d = np.linalg.norm(A_centered_2d, 'fro')
    A_normalized_2d = A_centered_2d / scale_2d if scale_2d != 0 else A_centered_2d
    
    # Normalize 3D
    A_centered_3d = A_3d - np.mean(A_3d, axis=0)
    scale_3d = np.linalg.norm(A_centered_3d, 'fro')
    A_normalized_3d = A_centered_3d / scale_3d if scale_3d != 0 else A_centered_3d
    
    return A_normalized_2d, A_normalized_3d

def procrustes_error(test_matrix, gt_matrix):
    """
    Calculates the optimal rotation to align test_matrix to gt_matrix using SVD,
    then returns the Frobenius norm of the difference (the error).
    """
    # 1. Calculate Covariance Matrix: C = Test^T * GT
    C = np.dot(test_matrix.T, gt_matrix)
    
    # Ground Truth Numpy SVD
    U_np, sigma_np, Vt_np = np.linalg.svd(C)
    R_np = np.dot(U_np, Vt_np)
    if np.linalg.det(R_np) < 0:
        Vt_np[-1, :] *= -1.0
        R_np = np.dot(U_np, Vt_np)
    
    test_rotated_np = np.dot(test_matrix, R_np)
    error_np = np.linalg.norm(gt_matrix - test_rotated_np, 'fro')
    
    # Jacobi SVD
    U_j, sigma_j, Vt_j = jacobi_svd(C)
    R_j = np.dot(U_j, Vt_j)
    if np.linalg.det(R_j) < 0:
        Vt_j[-1, :] *= -1.0
        R_j = np.dot(U_j, Vt_j)
    
    test_rotated_j = np.dot(test_matrix, R_j)
    error_j = np.linalg.norm(gt_matrix - test_rotated_j, 'fro')
    
    return error_np, error_j

def max_off_diagonal(S):
    max_val = 0.0
    p, q = 0, 0
    rows = S.shape[0]
    
    for i in range(rows):
        for j in range(i + 1, rows):
            current_abs = abs(S[i, j])  
            if current_abs > max_val:
                max_val = current_abs
                p = i
                q = j
    return max_val, p, q

def max_off_diagonal_numpy(S):
    # Isolate the upper triangle and take the absolute value
    upper_triangle = np.abs(np.triu(S, k=1))
    
    # Find the 2D coordinates of the maximum value
    p, q = np.unravel_index(np.argmax(upper_triangle), S.shape)
    
    # Get the actual maximum value using those coordinates
    max_value = upper_triangle[p, q]
    
    return max_value, p, q

def jacobi_svd(C, tolerance=1e-9, max_iter=100):
    n = C.shape[0]
    S = C.T @ C
    V = np.eye(n)
    
    
    for iter in range(max_iter):
        max_off_diag, p, q = max_off_diagonal_numpy(S)
        
        if max_off_diag < tolerance:
            break
        
        # Calculate the rotation angle (theta) to zero out S[p, q]
        if abs(S[p, p] - S[q, q] < 1e-15):
            theta = np.pi / 4.0
        else:
            theta = 0.5 * np.arctan(2 * S[p,q] / (S[p, p] - S[q, q]))
        
        # Create a 3x3 Identity Matrix (Givens Matrix)
        G = np.eye(n)
        G[p, p] = np.cos(theta)
        G[q, q] = np.cos(theta)
        G[p, q] = -np.sin(theta)
        G[q, p] = np.sin(theta)
        
        # Apply the rotation to S (forces S[p, q] to 0)
        S = G.T @ S @ G
        
        # Accumulate the rotation in V
        V = V @ G
     
    # Extract the diagonal elements (the eigenvalues)   
    eigenvalues = np.diag(S)
    # Clamping negative numbers to zero
    eigenvalues = np.maximum(eigenvalues, 0.0)
    
    # Get singular values
    sigma = np.sqrt(eigenvalues)
    
    U = np.zeros((n, n))
    for i in range(n):
        if sigma[i] > tolerance:
            U[:,i] = np.dot(C, V[:, i]) / sigma[i]
        else:
            U[i, i] = 1.0
            
    return U, sigma, V.T
        

if __name__ == "__main__":
    test_dir = "SVD_Check"
    
    pose_configs = {
        "Balancing Table Pose": {
            "test_folder": "balancing_table",
            "gt_2d_file": "balancing_table_gt_2d.npy",
            "gt_3d_file": "balancing_table_gt_3d.npy"
        },
        "Cobra Pose": {
            "test_folder": "cobra",
            "gt_2d_file": "cobra_gt_2d.npy",
            "gt_3d_file": "cobra_gt_3d.npy"
        },
        "Downward Facing Dog Pose": {
            "test_folder": "downward_facing_dog",
            "gt_2d_file": "downward_facing_dog_gt_2d.npy",
            "gt_3d_file": "downward_facing_dog_gt_3d.npy"
        },
        "One-Legged King Pigeon Pose": {
            "test_folder": "one_legged_king_pigeon",
            "gt_2d_file": "one_legged_king_pigeon_gt_2d.npy",
            "gt_3d_file": "one_legged_king_pigeon_gt_3d.npy"
        },
        "Tree Pose": {
            "test_folder": "tree",
            "gt_2d_file": "tree_gt_2d.npy",
            "gt_3d_file": "tree_gt_3d.npy"
        },
        "Triangle Pose": {
            "test_folder": "triangle",
            "gt_2d_file": "triangle_gt_2d.npy",
            "gt_3d_file": "triangle_gt_3d.npy"
        }
    }
    
    for pose_name, config in pose_configs.items():
        print(f"\n========================================")
        print(f"Validating: {pose_name}")
        
        # Load Ground Truths
        try:
            gt_2d_path = os.path.join(test_dir, config["test_folder"], config["gt_2d_file"])
            gt_3d_path = os.path.join(test_dir, config["test_folder"], config["gt_3d_file"])
            
            gt_2d = np.load(gt_2d_path)
            gt_3d = np.load(gt_3d_path)
            
        except FileNotFoundError:
            print(f"  -> Error: Could not find ground truth .npy files for {pose_name}.")
            continue
            
        # Get Test Images
        folder_path = os.path.join(test_dir, config["test_folder"])
        if not os.path.exists(folder_path):
            print(f"  -> Error: Could not find test folder {folder_path}")
            continue
            
        valid_extensions = ('.jpg', '.jpeg', '.png')
        test_images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        if not test_images:
            print(f"  -> No test images found in {folder_path}")
            continue
            
        for img_name in test_images:
            img_path = os.path.join(folder_path, img_name)
            test_2d, test_3d = get_normalized_pose_matrices(img_path)
            
            if test_2d is not None and test_3d is not None:
                error_2d, jacobi_err_2d = procrustes_error(test_2d, gt_2d)
                error_3d, jacobi_err_3d = procrustes_error(test_3d, gt_3d)
                
                print(f"\n  Image: {img_name}")
                print(f"    2D Alignment Error -> NumPy: {error_2d:.4f} | Jacobi: {jacobi_err_2d:.4f}")
                print(f"    3D Alignment Error -> NumPy: {error_3d:.4f} | Jacobi: {jacobi_err_3d:.4f}")
            else:
                print(f"\n  Image: {img_name} -> Failed to detect pose.")