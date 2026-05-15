import logging
import os
from PIL import Image
import cv2
import numpy as np


def clean_image(image_path, output_dir, binary_threshold_block_size=21, binary_threshold_c=8):
    """Cleans the stitched image by applying binarization.

    Args:
        image_path (_type_): The file path to the stitched image that needs to be cleaned.
        output_dir (_type_): The directory where the cleaned image will be saved.
        binary_threshold_block_size (_type_): The block size for adaptive thresholding (0-255).
        binary_threshold_c (_type_): The constant subtracted from the mean or weighted mean.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read the stitched image at: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for binarization
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, binary_threshold_block_size, binary_threshold_c)
    # Save the cleaned image
    cv2.imwrite(os.path.join(output_dir, "cleaned_sheet.png"), binary)
    logging.info(f"Cleaned image saved to: {os.path.join(output_dir, 'cleaned_sheet.png')}")

def generate_pdf(image_path, output_pdf_path, top_margin_px=80, system_spacing_px=80):
    """
    Slices a long stitched image into A4-proportioned chunks and saves them as a multi-page PDF.

    Args:
        image_path (_type_): The file path to the cleaned stitched image that needs to be converted to PDF.
        output_pdf_path (_type_): The file path where the final PDF will be saved.
        top_margin_px (_type_): The height of the white margin to add at the top of each page in pixels (default: 80).
        system_spacing_px (_type_): The number of pixels to insert as a margin between systems in pixels (default: 80).
    """
    img = cv2.imread(image_path)
    if img is None:
        logging.error("Could not load image for PDF generation.")
        return
    
    h, w, _ = img.shape
    
    # A4 is 210mm x 297mm. The ratio is 297/210=1.414
    a4_ratio = 297/210
    page_height_px = int(w * a4_ratio)
    max_chunk_height = page_height_px - top_margin_px

    pages = []
    current_y = 0
    # We want to ensure we find a gap reasonably close to the expected system spacing but not necessarily exactly at it (to account for variations in the video + safe padding)
    min_gap_to_look_for = int(system_spacing_px * 0.75)
    
    while current_y < h:
        end_row = current_y + max_chunk_height
        
        # We don't want to cut through staff lines. If the end_row isn't already in a gap, look upwards to find the nearest one.
        if end_row < h:
            # Look up to 25% up the page to find a gap
            search_limit = max(current_y + min_gap_to_look_for, end_row - int(max_chunk_height * 0.25))
            best_cut = end_row
            
            for y in range(end_row, search_limit, -1):
                # Cut if we find our margin gap
                if np.all(img[y-min_gap_to_look_for:y] == 255):
                    best_cut = y - (min_gap_to_look_for // 2)
                    break
            
            end_row = best_cut

        chunk = img[current_y:end_row, :]
        # Pad the top with white to create a margin on top of each page
        top_padding = np.ones((top_margin_px, w, 3), dtype=np.uint8) * 255
        chunk = np.vstack([top_padding, chunk])
        
        # Pad the bottom with white so every page remains exactly A4 sized
        if chunk.shape[0] < page_height_px:
            padding = np.ones((page_height_px - chunk.shape[0], w, 3), dtype=np.uint8) * 255
            chunk = np.vstack([chunk, padding])

        chunk_rgb = cv2.cvtColor(chunk, cv2.COLOR_BGR2RGB)
        pages.append(Image.fromarray(chunk_rgb))
        
        # Update the starting point for the next page
        current_y = end_row

    if pages:
        pages[0].save(
            output_pdf_path, 
            "PDF", 
            save_all=True, 
            append_images=pages[1:], 
            resolution=100.0
        )
        logging.info(f"PDF generated and saved to: {output_pdf_path}")