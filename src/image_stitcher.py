import logging
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from src import config

def stitch_images(frames_dir, output_dir, method="system_hard_cut", system_spacing=80):
    """Stitches the extracted frames together into a single continuous image.

    Args:
        frames_dir (_type_): The directory containing the extracted frames to be stitched.
        output_dir (_type_): The directory where the final stitched image will be saved.
        method (_type_): The method to use for stitching (default: "system_hard_cut").
        system_spacing (_type_): The number of pixels to insert as a margin between systems (default: 80).
    """
    all_files = glob.glob(os.path.join(frames_dir, "*.png"))
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
        final_page = stitch_system_hard_cut(frame_files, system_spacing=system_spacing)
    else:
        raise ValueError(f"Unknown stitching method: {method}")
    
    cv2.imwrite(os.path.join(output_dir, "stitched_sheet.png"), final_page)
    logging.info(f"Saved to: {os.path.join(output_dir, 'stitched_sheet.png')}")


def extract_frame_number(filepath):
    """Safely attempts to extract the integer from a 'frame_X.png' filename."""
    filename = os.path.basename(filepath)
    try:
        num_str = filename.split('_')[1].split('.')[0]
        return int(num_str)
    except (IndexError, ValueError):
        return None

def stitch_system_hard_cut(frame_files, system_spacing=80):
    """
    Stitches images together vertically.

    Args:
        frame_files: A list of file paths to the frames to be stitched.
        system_spacing: The number of pixels to insert as a margin between systems.
    """
    if not frame_files:
        logging.error("No frames provided to stitch.")
        return None

    lines_to_stack = []
    first_frame = cv2.imread(frame_files[0])
    first_frame = auto_trim_vertical(first_frame)
    lines_to_stack.append(first_frame)

    # Pre-calculate the white margin
    _, w, c = first_frame.shape
    white_margin = np.ones((system_spacing, w, c), dtype=np.uint8) * 255

    for i in tqdm(range(1, len(frame_files)), desc="Stitching frames"):
        logging.debug(f"Adding line {i+1}...")
        frame = cv2.imread(frame_files[i])
        frame = auto_trim_vertical(frame)
        lines_to_stack.append(white_margin)
        lines_to_stack.append(frame)

    # Stack everything top-to-bottom
    final_page = np.vstack(lines_to_stack)
    logging.info(f"Success! Assembled {len(frame_files)} lines into a single page.")
    
    return final_page

def auto_trim_vertical(frame, ink_threshold=200, safety_pad=5):
    """Finds the actual sheet music ink and slices away the excess white space at the top and bottom of the frame.

    Args:
        frame: The input image frame to be trimmed.
        ink_threshold: The pixel intensity threshold below which a pixel is considered "ink" (default: 200).
        safety_pad: The number of pixels to add as a safety margin (default: 5).

    Returns:
        The trimmed image frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find all rows that contain at least one dark pixel
    ink_rows = np.any(gray < ink_threshold, axis=1)
    ink_indices = np.where(ink_rows)[0]
    
    # Safety check: If the frame is completely blank, just return it
    if len(ink_indices) == 0:
        return frame
        
    # Find the top and bottom of the music
    top_ink = ink_indices[0]
    bottom_ink = ink_indices[-1]
    
    top_cut = max(0, top_ink - safety_pad)
    bottom_cut = min(frame.shape[0], bottom_ink + safety_pad)
    
    return frame[top_cut:bottom_cut, :]