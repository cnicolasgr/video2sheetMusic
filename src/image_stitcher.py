import logging
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from src import config

def stitch_images(frames_dir, output_file_path, method="system_hard_cut", margin_height=80, bottom_crop=0):
    """Stitches the extracted frames together into a single continuous image.

    Args:
        frames_dir (_type_): The directory containing the extracted frames to be stitched.
        output_file_path (_type_): The file path where the final stitched image will be saved.
        method (_type_): The method to use for stitching (default: "system_hard_cut").
        margin_height (_type_): The height of the margin to be added between systems.
        bottom_crop (_type_): The number of pixels to crop from the bottom of each frame.
    """
    all_files = glob.glob(os.path.join(frames_dir, "*.jpg"))
    valid_files = []
    for f in all_files:
        if extract_frame_number(f) is not None:
            valid_files.append(f)
        else:
            logging.warning(f"Skipping badly formatted file: {os.path.basename(f)}")

    if not valid_files:
        logging.error("No valid frame files found to stitch.")
        return None
    
    frame_files = sorted(valid_files, key=extract_frame_number)
    
    if method == "system_hard_cut":
        final_page = stitch_system_hard_cut(frame_files, margin_height=margin_height, bottom_crop=bottom_crop)
    else:
        raise ValueError(f"Unknown stitching method: {method}")
    
    cv2.imwrite(output_file_path, final_page)
    logging.info(f"Saved to: {output_file_path}")

def stitch_system_hard_cut(frame_files, margin_height=80, bottom_crop=0):
    """
    Stitches images together vertically.
    - margin_height > 0: Adds white space between frames.
    - margin_height < 0: Crops pixels from the TOP of the incoming frame.
    - bottom_crop > 0: Crops pixels from the BOTTOM of every frame.
    """
    if not frame_files:
        logging.error("No frames provided to stitch.")
        return None

    lines_to_stack = []
    
    first_frame = cv2.imread(frame_files[0])
    
    # Apply bottom crop to the first frame
    if bottom_crop > 0:
        if bottom_crop < first_frame.shape[0]:
            first_frame = first_frame[:-bottom_crop, :] 
        else:
            logging.warning("Bottom crop is larger than the first frame's height!")
            
    lines_to_stack.append(first_frame)
    
    # Pre-calculate the white margin (using the width of the first frame)
    _, w, c = first_frame.shape
    if margin_height > 0:
        white_margin = np.ones((margin_height, w, c), dtype=np.uint8) * 255

    for i in tqdm(range(1, len(frame_files)), desc="Stitching frames"):
        logging.debug(f"Adding line {i+1}...")
        frame = cv2.imread(frame_files[i])
        
        if bottom_crop > 0:
            if bottom_crop < frame.shape[0]:
                # Slice off the last 'bottom_crop' rows
                frame = frame[:-bottom_crop, :]
            else:
                logging.warning(f"Frame {i+1} is too short to bottom-crop {bottom_crop} pixels!")

        if margin_height > 0:
            lines_to_stack.append(white_margin)
            lines_to_stack.append(frame)
            
        elif margin_height < 0:
            top_crop_amount = abs(margin_height)
            if top_crop_amount < frame.shape[0]:
                # Slice off the first 'top_crop_amount' rows
                frame = frame[top_crop_amount:, :] 
            else:
                logging.warning(f"Frame {i+1} is too short to top-crop {top_crop_amount} pixels!")
            lines_to_stack.append(frame)
            
        else:
            lines_to_stack.append(frame)

    # Stack everything top-to-bottom
    final_page = np.vstack(lines_to_stack)
    logging.info(f"Success! Assembled {len(frame_files)} lines into a single page.")
    
    return final_page

def extract_frame_number(filepath):
    """Safely attempts to extract the integer from a 'frame_X.jpg' filename."""
    filename = os.path.basename(filepath)
    try:
        num_str = filename.split('_')[1].split('.')[0]
        return int(num_str)
    except (IndexError, ValueError):
        return None