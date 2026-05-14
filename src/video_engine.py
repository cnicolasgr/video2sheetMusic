import logging
import cv2
import numpy as np
from yt_dlp import YoutubeDL
from skimage.metrics import structural_similarity as ssim
from src import config
import os
from tqdm import tqdm


def download_video(video_url, output_dir):
    """Downloads a YouTube video using yt-dlp and saves it to the specified output directory.

    Args:
        video_url (_type_): The URL of the YouTube video to download.
        output_dir (_type_): The directory where the downloaded video will be saved.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'quiet': True,
        'noprogress': True,
        'logger': logging.getLogger(),
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return ydl.prepare_filename(ydl.extract_info(video_url, download=False))
        
        
def extract_frames(video_path, output_dir, skip_rate, ssim_threshold):
    """Extracts frames from a video file and saves them to the specified output directory.

    Args:
        video_path (_type_): The path to the video file.
        output_dir (_type_): The directory where the extracted frames will be saved.
        skip_rate (_type_): The number of frames to skip between checks for significant changes.
        ssim_threshold (_type_): The SSIM threshold for determining significant changes between frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error: Could not open video.")
        exit()

    # Detect the sheet music area and crop the frame to focus on it
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    success, mid_frame = cap.read()
    if not success:
        logging.error("Could not read the middle frame of the video.")
        return
    x, y, w, h = auto_crop_image(mid_frame)

    # Read the first frame to initialize the previous frame for comparison
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, prev_frame = cap.read()
    if not success:
        logging.error("Could not read the first frame of the video.")
        return

    prev_frame = prev_frame[y:y+h, x:x+w]

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/frame_1.jpg", prev_frame)
    
    saved_frame_count = 1
    frame_count = 1

    with tqdm(total=total_frames, initial=1, desc="Analyzing Video") as pbar:
        # Loop through the video frame-by-frame
        while True:
            success, curr_frame = cap.read()
            if not success:
                logging.debug("Reached the end of the video.")
                pbar.update(total_frames - pbar.n)
                break

            frame_count += 1
            pbar.update(1)
            if frame_count % skip_rate != 0:
                continue
            if np.mean(curr_frame) < 10.0:
                logging.debug(f"Skipping frame {frame_count} (Black screen detected)")
                continue
            
            curr_frame = curr_frame[y:y+h, x:x+w]
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Use structural similarity index (SSIM) to check for significant changes between frames
            ssim_const = ssim(prev_gray, curr_gray, data_range=255)
            if ssim_const < ssim_threshold:
                saved_frame_count += 1
                cv2.imwrite(f"{output_dir}/frame_{saved_frame_count}.jpg", curr_gray)
                logging.debug(f"Saved frame {saved_frame_count} (SSIM: {ssim_const:.4f})")
                prev_gray = curr_gray

    # Clean up
    cap.release()

    logging.info(f"Processed {frame_count} frames.")
    return output_dir

def auto_crop_image(image):
    """Automatically crops the image to focus on the sheet music area.

    Args:
        image (_type_): The image to be cropped.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logging.warning("No contours found for cropping. Fallback to user-defined ROI.")
        return prompt_sheet_roi(image)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    biggest_contour = contours[0]
    x, y, w, h = cv2.boundingRect(biggest_contour)

    frame_area = image.shape[0] * image.shape[1]
    detected_area = w * h

    if detected_area / frame_area < 0.1:
        logging.warning("Detected area is too small. Fallback to user-defined ROI.")
        return prompt_sheet_roi(image)

    # Add a 10-pixel padding so we don't catch the video background
    padding = 10
    x = x + padding
    y = y + padding
    w = w - padding
    h = h - padding

    return x, y, w, h

def prompt_sheet_roi(frame):
    """Prompts the user to crop the frame to focus on the sheet music area.

    Args:
        frame (_type_): The frame image to be cropped.
    """
    # Display the frame and allow the user to select a region of interest (ROI)
    r = cv2.selectROI("Select Sheet Music Area", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    return r

def setup_directories(frames_dir):
    """Sets up the necessary directories for the video processing pipeline."""

    # Create necessary directories if they don't exist
    os.makedirs(config.INPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Clear the output and temporary directories
    for filename in os.listdir(config.INPUT_VIDEOS_DIR):
        file_path = os.path.join(config.INPUT_VIDEOS_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error(f"Error while clearing input videos directory: {e}")


    for filename in os.listdir(frames_dir):
        file_path = os.path.join(frames_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error(f"Error while clearing temporary frames directory: {e}")