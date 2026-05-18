import numpy as np
import matplotlib.pyplot as plt
import json
import os

# --- Configuration ---
GT_DIR = "GT_Data" # Or wherever your files are saved
POSE_NAME = "triangle" # Change this to test different poses

# The 12 core structural joints (mapped to indices 0-11 in your matrix)
# 0: L Shoulder, 1: R Shoulder
# 2: L Elbow,    3: R Elbow
# 4: L Wrist,    5: R Wrist
# 6: L Hip,      7: R Hip
# 8: L Knee,     9: R Knee
# 10: L Ankle,   11: R Ankle

CONNECTIONS = [
    (0, 1),           # Shoulders
    (0, 2), (2, 4),   # Left Arm
    (1, 3), (3, 5),   # Right Arm
    (0, 6), (1, 7),   # Torso (Shoulders to Hips)
    (6, 7),           # Hips
    (6, 8), (8, 10),  # Left Leg
    (7, 9), (9, 11)   # Right Leg
]

def validate_pose(pose_name):
    npy_path = os.path.join(GT_DIR, pose_name, f"{pose_name}_gt.npy")
    json_path = os.path.join(GT_DIR, pose_name, f"{pose_name}_gt_angles.json")
    
    # 1. Verify and Load JSON Angles
    print(f"\n--- VALIDATING: {pose_name.upper()} ---")
    try:
        with open(json_path, 'r') as f:
            angles = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {json_path}")
        return

    # 2. Verify and Load Matrix
    try:
        matrix = np.load(npy_path)
        num_dims = matrix.shape[1]
        print(f"\n📏 Matrix Shape: {matrix.shape} (Detected as {num_dims}D Pose)")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {npy_path}")
        return

    is_2d = (num_dims == 2)

    # 3. Render the Dashboard (Wide Figure for Side-by-Side)
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Ground Truth Validation: {pose_name.replace('_', ' ').title()}", 
                 fontsize=16, fontweight='bold')
    
    # --- LEFT PANEL: SKELETON RENDERER ---
    if is_2d:
        ax = fig.add_subplot(121) # 1 row, 2 columns, 1st plot
        
        # Extract X and Y (Invert Y because MediaPipe Y goes down)
        xs = matrix[:, 0]
        ys = -matrix[:, 1]
        
        ax.scatter(xs, ys, c='red', s=50, label='Joints')

        for start_idx, end_idx in CONNECTIONS:
            start_pt = matrix[start_idx]
            end_pt = matrix[end_idx]
            if np.all(start_pt == 0) or np.all(end_pt == 0):
                continue 
            ax.plot([xs[start_idx], xs[end_idx]], [ys[start_idx], ys[end_idx]], 'b-', linewidth=2)

        labels = ["LS", "RS", "LE", "RE", "LW", "RW", "LH", "RH", "LK", "RK", "LA", "RA"]
        for i, txt in enumerate(labels):
            if not np.all(matrix[i] == 0):
                ax.text(xs[i] + 0.02, ys[i], txt, size=8, zorder=1, color='k')

        ax.set_title(f"2D Geometry")
        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Y (Up/Down)')
        ax.set_aspect('equal', adjustable='datalim') 

    else:
        ax = fig.add_subplot(121, projection='3d') # 1 row, 2 columns, 1st plot
        
        # Extract X, Y, Z
        xs = matrix[:, 0]
        ys = matrix[:, 2] 
        zs = -matrix[:, 1] 

        ax.scatter(xs, ys, zs, c='red', s=50, label='Joints')

        for start_idx, end_idx in CONNECTIONS:
            start_pt = matrix[start_idx]
            end_pt = matrix[end_idx]
            if np.all(start_pt == 0) or np.all(end_pt == 0):
                continue 
            ax.plot([xs[start_idx], xs[end_idx]], 
                    [ys[start_idx], ys[end_idx]], 
                    [zs[start_idx], zs[end_idx]], 'b-', linewidth=2)

        labels = ["LS", "RS", "LE", "RE", "LW", "RW", "LH", "RH", "LK", "RK", "LA", "RA"]
        for i, txt in enumerate(labels):
            if not np.all(matrix[i] == 0):
                ax.text(xs[i], ys[i], zs[i], txt, size=8, zorder=1, color='k')

        ax.set_title(f"3D Geometry")
        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Up/Down)')
        
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- RIGHT PANEL: BIOMECHANICAL ANGLES ---
    ax_text = fig.add_subplot(122) # 1 row, 2 columns, 2nd plot
    ax_text.axis('off') # Hide the grid and axes for a clean text look
    
    ax_text.text(0.1, 0.95, "Averaged Biomechanical Angles", 
                 fontsize=14, fontweight='bold', transform=ax_text.transAxes)
    ax_text.text(0.1, 0.90, "-"*40, fontsize=12, transform=ax_text.transAxes)
    
    y_pos = 0.82
    for joint, angle in angles.items():
        if angle is not None:
            # Format nicely: "Left Elbow: 145.2°"
            text_str = f"{joint}: {angle:>6}°"
            color = 'black'
            fontweight = 'normal'
        else:
            text_str = f"{joint}: [HIDDEN]"
            color = 'red'
            fontweight = 'bold'
            
        ax_text.text(0.15, y_pos, text_str, fontsize=12, color=color, 
                     fontweight=fontweight, fontfamily='monospace', transform=ax_text.transAxes)
        y_pos -= 0.08 # Move down for the next line

    plt.tight_layout()
    print("\n✅ Launching Presentation Dashboard! Close the window to exit.")
    plt.show()

if __name__ == "__main__":
    validate_pose(POSE_NAME)