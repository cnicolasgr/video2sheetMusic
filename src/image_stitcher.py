import cv2
import numpy as np
import os
import glob

from src import config

def stitch_images(frames_dir, output_dir, method="system_hard_cut", margin_height=80):
    """Stitches the extracted frames together into a single continuous image.

    Args:
        frames_dir (_type_): The directory containing the extracted frames to be stitched.
        output_dir (_type_): The directory where the final stitched image will be saved.
        method (_type_): The method to use for stitching (default: "system_hard_cut").
        margin_height (_type_): The height of the margin to be added between systems.
    """
            
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")), 
                         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    if not frame_files:
        print("Error: No frames found to stitch.")
        return None
    
    if method == "system_hard_cut":
        final_page = stitch_system_hard_cut(frame_files, margin_height=margin_height)
    else:
        raise ValueError(f"Unknown stitching method: {method}")
    
    final_output_path = os.path.join(output_dir, "final_sheet.jpg")
    cv2.imwrite(final_output_path, final_page)
    print(f"Saved to: {final_output_path}")

def stitch_system_hard_cut(frame_files, margin_height=80):
    """Stitches images together using a simple hard cut approach, stacking them vertically with a white margin in between.
    Args:
        frame_files (_type_): A list of file paths to the frames to be stitched.
        margin_height (_type_): The height of the margin to be added between systems.
    """

    lines_to_stack = []
    
    sample_frame = cv2.imread(frame_files[0])
    _, w, c = sample_frame.shape
    
    white_margin = np.ones((margin_height, w, c), dtype=np.uint8) * 255

    for i, file_path in enumerate(frame_files):
        print(f"Adding line {i+1}...")
        frame = cv2.imread(file_path)
        lines_to_stack.append(frame)
        lines_to_stack.append(white_margin)

    final_page = np.vstack(lines_to_stack)
    print(f"Success! Assembled {len(frame_files)} lines into a single page.")
    return final_page

def setup_directories():
    """Sets up the necessary directories for the video processing pipeline."""

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # Clear the output directory
    for filename in os.listdir(config.OUTPUT_DIR):
        file_path = os.path.join(config.OUTPUT_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error while clearing output directory: {e}")