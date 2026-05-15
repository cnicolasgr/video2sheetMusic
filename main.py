import argparse
import logging
import os
import sys

from src.config import Config
from src.image_cleaner import clean_image, generate_pdf
from src.image_stitcher import stitch_images
from src.video_engine import download_video, extract_frames


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
        "--output_dir", "-o",
        type=str, 
        default=f"{Config.paths.OUTPUT}", 
        help="Path to save the final output (default: data/output/)."
    )

    parser.add_argument(
        "--skip-frames", "-s",
        type=int,
        default=Config.video.FRAME_SKIP_RATE,
        help=f"Number of frames to skip between checks for significant changes (default: {Config.video.FRAME_SKIP_RATE})."
    )

    parser.add_argument(
        "--ssim-threshold", "-t",
        type=float,
        default=Config.video.SSIM_THRESHOLD,
        help=f"SSIM threshold for determining significant changes between frames (default: {Config.video.SSIM_THRESHOLD})."
    )

    parser.add_argument(
        "--system-spacing", "-sp",
        type=int,
        default=Config.stitch.SYSTEM_SPACING,
        help=f"Number of pixels to insert as a margin between systems in pixels (default: {Config.stitch.SYSTEM_SPACING})."
    )

    
    args = parser.parse_args()

    reset(Config.paths.INPUT_VIDEOS, Config.paths.TEMP_FRAMES, args.output_dir)

    logging.info(f"Starting extraction process for: {args.url}")

    try:
        # Phase 1: Video Acquisition & Frame Extraction
        logging.debug("[1/4] Downloading video & extracting unique frames...")
        video_path = download_video(args.url, output_dir=Config.paths.INPUT_VIDEOS)
        extract_frames(video_path, output_dir=Config.paths.TEMP_FRAMES, skip_rate=args.skip_frames, ssim_threshold=args.ssim_threshold)
        
        # Phase 2: Image Stitching
        logging.info("[2/4] Stitching frames into a continuous panorama...")
        stitch_images(Config.paths.TEMP_FRAMES, output_dir=args.output_dir, method=Config.stitch.METHOD, system_spacing=args.system_spacing)
        
        # Phase 3: Image Cleanup (Binarization)
        logging.info("[3/4] Cleaning the stitched image...")
        clean_image(os.path.join(args.output_dir, "stitched_sheet.png"), output_dir=args.output_dir, binary_threshold_block_size=Config.clean.BLOCK_SIZE, binary_threshold_c=Config.clean.CONSTANT_C)

        # Phase 4: PDF Generation
        logging.info("[4/4] Generating PDF...")
        sheet_name = os.path.basename(video_path).rsplit('.', 1)[0]
        generate_pdf(os.path.join(args.output_dir, "cleaned_sheet.png"), os.path.join(args.output_dir, f"{sheet_name}.pdf"), top_margin_px=Config.clean.PDF_TOP_MARGIN_PX, system_spacing_px=args.system_spacing)

        print(f"\nSuccess! Final output saved to: {args.output_dir}")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        sys.exit(1)

def reset(input_videos_dir, frames_dir, output_dir):
    """Utility function to clear out old data from previous runs.
    
    Args:
        input_videos_dir (_type_): Directory where input videos are stored.
        frames_dir (_type_): Directory where extracted frames are stored.
        output_dir (_type_): Directory where final outputs are stored.
    """
    # Reset directories
    for dir_path in [frames_dir, output_dir, input_videos_dir]:
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, f))
        else:
            os.makedirs(dir_path)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    logging.basicConfig(
        filename='data/main_log.log',
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    main()