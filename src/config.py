from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class PathConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INPUT_VIDEOS: Path = DATA_DIR / "input_video"
    TEMP_FRAMES: Path = DATA_DIR / "temp_frames"
    OUTPUT: Path = DATA_DIR / "output"

@dataclass(frozen=True)
class VideoConfig:
    SSIM_THRESHOLD: float = 0.88
    FRAME_SKIP_RATE: int = 5

@dataclass(frozen=True)
class StitchConfig:
    METHOD: str = "system_hard_cut"
    SYSTEM_SPACING: int = 80

@dataclass(frozen=True)
class CleanConfig:
    BLOCK_SIZE: int = 21
    CONSTANT_C: int = 8
    PDF_TOP_MARGIN_PX: int = 80

class Config:
    paths = PathConfig()
    video = VideoConfig()
    stitch = StitchConfig()
    clean = CleanConfig()