import os
import shutil

project_root = r'e:\-VisionExtract-Isolation-from-Images-using-Image-Segmentation'
target_dir = os.path.join(project_root, 'docs', 'images')
source_path = r'C:\Users\BISWAJEET KUMAR\.gemini\antigravity\brain\47692a61-dae7-4062-8abf-8df7ac7972c9\vision_extract_banner_1774587551882.png'
dest_path = os.path.join(target_dir, 'banner.png')

print(f"Creating directory: {target_dir}")
os.makedirs(target_dir, exist_ok=True)

if os.path.exists(source_path):
    print(f"Copying {source_path} to {dest_path}")
    shutil.copy2(source_path, dest_path)
    print("Success!")
else:
    print(f"Error: Source file not found at {source_path}")
