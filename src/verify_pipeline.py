import os
import sys

# Ensure current directory is in the path
sys.path.append(os.path.dirname(__file__))

from inference import VisionExtractPipeline

def verify():
    print("--- Starting Verification ---")
    
    # Initialize pipeline
    # We use a dummy model as we haven't trained one in this session
    # The goal is to verify the logic and file operations
    pipeline = VisionExtractPipeline()
    
    # Define test and output folders
    root_dir = os.path.dirname(os.path.dirname(__file__))
    test_folder = os.path.join(root_dir, "test_images")
    output_folder = os.path.join(root_dir, "outputs")
    
    print(f"Testing folder: {test_folder}")
    
    if os.path.exists(test_folder) and os.listdir(test_folder):
        pipeline.batch_inference(test_folder)
    else:
        print(f"Error: {test_folder} is empty or missing.")
        
    if os.path.exists(output_folder) and os.listdir(output_folder):
        print(f"Success: Outputs generated in {output_folder}")
        for out in os.listdir(output_folder):
            print(f"  - {out}")
    else:
        print(f"Failure: No outputs found in {output_folder}")
        
    print("--- Verification Finished ---")

if __name__ == "__main__":
    verify()
