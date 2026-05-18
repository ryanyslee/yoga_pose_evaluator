import urllib.request
import os

def download_model():
    # Official Google MediaPipe URL for the Heavy Pose Landmarker
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    file_path = os.path.join("models", "pose_landmarker_heavy.task")
    
    if os.path.exists(file_path):
        print("✅ Model already exists!")
        return

    print("Downloading pose_landmarker_heavy.task (this may take a minute)...")
    urllib.request.urlretrieve(url, file_path)
    print(f"✅ Successfully downloaded to {file_path}")

if __name__ == "__main__":
    download_model()