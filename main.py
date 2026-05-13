import argparse
import os
import sys

from src import config
from src.video_engine import setup_directories

try:
    from src.video_engine import download_video, extract_frames
    from src.image_stitcher import stitch_images
    from src.image_cleaner import clean_image
    from src.omr_wrapper import process_omr
except ImportError as e:
    print(f"Note: Some modules are not yet implemented. Details: {e}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extract and reconstruct sheet music from scrolling YouTube videos."
    )
    
    parser.add_argument(
        "url", 
        type=str, 
        help="The YouTube URL of the sheet music video."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str, 
        default="data/output/final_sheet.pdf", 
        help="Path and filename to save the final output (default: data/output/final_sheet.pdf)."
    )

    parser.add_argument(
        "--frames-dir", "-f",
        type=str, 
        default=config.TEMP_FRAMES_DIR, 
        help=f"Path to frames directory (default: {config.TEMP_FRAMES_DIR})."
    )

    parser.add_argument(
        "--skip-frames", "-s",
        type=int,
        default=config.FRAME_SKIP_RATE,
        help=f"Number of frames to skip between checks for significant changes (default: {config.FRAME_SKIP_RATE})."
    )

    parser.add_argument(
        "--ssim-threshold", "-t",
        type=float,
        default=config.SSIM_THRESHOLD,
        help=f"SSIM threshold for determining significant changes between frames (default: {config.SSIM_THRESHOLD})."
    )

    parser.add_argument(
        "phases",
        nargs="*",
        choices=["download", "stitch", "clean", "omr"],
        help="Optional: Specify which phases to run (default: all)."
    )
    
    args = parser.parse_args()
    youtube_url = args.url
    output_path = args.output
    skip_frames = args.skip_frames
    ssim_threshold = args.ssim_threshold
    frames_dir = args.frames_dir
    phases = args.phases

    print(f"Starting extraction process for: {youtube_url}")

    try:
        if not phases or "download" in phases:
            # Phase 1: Video Acquisition & Frame Extraction
            print("\n[1/4] Downloading video & extracting unique frames...")
            setup_directories(frames_dir)
            video_path = download_video(youtube_url, output_dir=config.INPUT_VIDEOS_DIR)
            frames_dir = extract_frames(video_path, output_dir=frames_dir, skip_rate=skip_frames, ssim_threshold=ssim_threshold)
        
        if not phases or "stitch" in phases:
            # Phase 2: Image Stitching
            print("[2/4] Stitching frames into a continuous panorama...")
            stitched_image_path = stitch_images(frames_dir, output_dir=config.OUTPUT_DIR, method=config.STITCHING_METHOD, margin_height=config.MARGIN_HEIGHT)
        
        if not phases or "clean" in phases:
            # Phase 3: Image Cleanup (Binarization & Deskewing)
            print("[3/4] Cleaning and deskewing the stitched image...")
            # cleaned_image_path = clean_image(stitched_image_path, output_dir=config.OUTPUT_DIR)

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()