import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the main data folders
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_VIDEOS_DIR = os.path.join(DATA_DIR, "input_videos")
TEMP_FRAMES_DIR = os.path.join(DATA_DIR, "temp_frames")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")


# Structural Similarity Index (SSIM) threshold.
# 1.0 = identical. 0.85 to 0.90 is usually ideal
SSIM_THRESHOLD = 0.88

# Check every Nth frame to speed up processing.
FRAME_SKIP_RATE = 5


# Image stitching parameters
STITCHING_METHOD = "system_hard_cut"  # Options: "system_hard_cut"
MARGIN_HEIGHT = 80  # Height of the white margin between systems in pixels