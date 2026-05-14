import logging
import os

import cv2


def clean_image(image_path, output_file_path):
    """Cleans the stitched image by applying binarization.

    Args:
        image_path (_type_): The file path to the stitched image that needs to be cleaned.
        output_file_path (_type_): The file path where the cleaned image will be saved.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read the stitched image at: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for binarization
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 21, 8)
    # Save the cleaned image
    cv2.imwrite(output_file_path + ".jpg", binary)
    logging.info(f"Cleaned image saved to: {output_file_path}.jpg")

def setup_directories(output_dir):
    """Sets up the necessary directories for the video processing pipeline."""

    os.makedirs(output_dir, exist_ok=True)
    # Clear the output directory
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error(f"Error while clearing output directory: {e}")