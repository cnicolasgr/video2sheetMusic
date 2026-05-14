import argparse
import logging
import sys

from src import config
from src.image_cleaner import clean_image
from src.image_stitcher import stitch_images
from src.video_engine import download_video, extract_frames, setup_directories


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
        default=f"{config.OUTPUT_DIR}/final_sheet", 
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
        "--vertical-margin", "-m",
        type=int,
        default=config.MARGIN_HEIGHT,
        help=f"Height of the vertical margin between systems in pixels (default: {config.MARGIN_HEIGHT})."
    )

    parser.add_argument(
        "--vertical-bottom-crop", "-bc",
        type=int,
        default=config.BOTTOM_CROP,
        help=f"Number of pixels to crop from the BOTTOM of each frame (default: {config.BOTTOM_CROP})."
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
    margin_height = args.vertical_margin
    bottom_crop = args.vertical_bottom_crop
    frames_dir = args.frames_dir
    stitched_image_path = frames_dir + "/stitched_sheet.jpg"
    phases = args.phases

    logging.info(f"Starting extraction process for: {youtube_url}")

    try:
        if not phases or "download" in phases:
            # Phase 1: Video Acquisition & Frame Extraction
            logging.debug("[1/4] Downloading video & extracting unique frames...")
            setup_directories(frames_dir)
            video_path = download_video(youtube_url, output_dir=config.INPUT_VIDEOS_DIR)
            frames_dir = extract_frames(video_path, output_dir=frames_dir, skip_rate=skip_frames, ssim_threshold=ssim_threshold)
        
        if not phases or "stitch" in phases:
            # Phase 2: Image Stitching
            logging.info("[2/4] Stitching frames into a continuous panorama...")
            stitch_images(frames_dir, output_file_path=stitched_image_path, method=config.STITCHING_METHOD, margin_height=margin_height, bottom_crop=bottom_crop)
        
        if not phases or "clean" in phases:
            # Phase 3: Image Cleanup (Binarization)
            logging.info("[3/4] Cleaning the stitched image...")
            setup_directories(output_path.rsplit('/', 1)[0])
            clean_image(stitched_image_path, output_file_path=output_path)

        print(f"\n✅ Success! Final output saved to: {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(
        filename='data/main_log.log',
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    main()