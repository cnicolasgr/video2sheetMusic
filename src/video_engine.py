import cv2
from yt_dlp import YoutubeDL
from skimage.metrics import structural_similarity as ssim
from src import config
import os


def download_video(video_url, output_dir):
    """Downloads a YouTube video using yt-dlp and saves it to the specified output directory.

    Args:
        video_url (_type_): The URL of the YouTube video to download.
        output_dir (_type_): The directory where the downloaded video will be saved.
    """
    with YoutubeDL({'outtmpl': f'{output_dir}/%(title)s.%(ext)s'}) as ydl:
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
        print("Error: Could not open video.")
        exit()

    # Read the first frame to initialize the previous frame for comparison
    success, prev_frame = cap.read()
    if not success:
        print("Could not read the first frame of the video.")
        return

    # Allow user to define the region of interest (ROI) for the sheet music area
    x, y, w, h = prompt_sheet_roi(prev_frame)
    prev_frame = prev_frame[y:y+h, x:x+w]

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/frame_1.jpg", prev_frame)
    
    saved_frame_count = 1
    frame_count = 1

    # Loop through the video frame-by-frame
    while True:
        success, curr_frame = cap.read()
        if not success:
            print("Reached the end of the video.")
            break

        frame_count += 1
        if frame_count % skip_rate != 0:
            continue

        curr_frame = curr_frame[y:y+h, x:x+w]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Use structural similarity index (SSIM) to check for significant changes between frames
        ssim_const = ssim(prev_gray, curr_gray, data_range=255)
        if ssim_const < ssim_threshold:
            saved_frame_count += 1
            cv2.imwrite(f"{output_dir}/frame_{saved_frame_count}.jpg", curr_gray)
            print(f"Saved frame {saved_frame_count} (SSIM: {ssim_const:.4f})")
            prev_gray = curr_gray

    # Clean up
    cap.release()

    print(f"Processed {frame_count} frames.")
    return output_dir

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
            print(f"Error while clearing input videos directory: {e}")


    for filename in os.listdir(frames_dir):
        file_path = os.path.join(frames_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error while clearing temporary frames directory: {e}")