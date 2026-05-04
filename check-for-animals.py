#!/usr/bin/env python3
import argparse
import json
import logging
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from megadetector.detection.run_detector_batch import load_and_run_detector_batch
    MEGADETECTOR_AVAILABLE = True
except ImportError:
    MEGADETECTOR_AVAILABLE = False

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.mts', '.m4v'}
FFMPEG_TIMEOUT = 600  # 10 minutes max per video extraction
SPECIESNET_TIMEOUT = 1200  # 20 minutes max per speciesnet run

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging with appropriate level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def sanitize_name(name: str) -> str:
    """Convert a filename to a safe filesystem name."""
    if not name:
        return 'video'
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
    return safe.strip('_') or 'video'


def find_videos(input_dir: Path) -> List[Path]:
    """Find all video files in the input directory."""
    return sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ])


def extract_frames(video_path: Path, frames_dir: Path, every_n_seconds: float,
                   frame_quality: int = 2, resize_width: Optional[int] = None,
                   hwaccel: Optional[str] = None) -> List[Path]:
    """Extract frames from video at specified interval using ffmpeg."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / 'frame_%06d.jpg')

    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']

    # Add hardware acceleration if specified
    if hwaccel:
        if hwaccel == 'videotoolbox':  # macOS
            cmd.extend(['-hwaccel', 'videotoolbox'])
        elif hwaccel == 'cuda':  # NVIDIA GPU
            cmd.extend(['-hwaccel', 'cuda'])

    cmd.extend(['-i', str(video_path)])

    # Build video filter: fps + optional resize
    vf_parts = [f'fps=1/{every_n_seconds}']
    if resize_width:
        vf_parts.append(f'scale={resize_width}:-1')  # Maintain aspect ratio

    cmd.extend(['-vf', ','.join(vf_parts)])
    cmd.extend(['-q:v', str(frame_quality), pattern])

    logger.debug(f"Extracting frames from {video_path.name} with interval {every_n_seconds}s"
                 f"{f', resize={resize_width}px' if resize_width else ''}"
                 f"{f', hwaccel={hwaccel}' if hwaccel else ''}")
    try:
        subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after {FFMPEG_TIMEOUT}s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.strip()}") from e

    frames = sorted(frames_dir.glob('*.jpg'))
    logger.debug(f"Extracted {len(frames)} frames")
    return frames


def run_megadetector(frames_dir: Path, output_json: Path, detection_threshold: float = 0.2) -> Dict[str, Any]:
    """Run MegaDetector on extracted frames to detect animals.

    Returns:
        Dict with detection results in MegaDetector JSON format
    """
    if not MEGADETECTOR_AVAILABLE:
        raise RuntimeError("MegaDetector is not installed. Install with: pip install megadetector")

    logger.debug(f"Running MegaDetector on {frames_dir} (threshold={detection_threshold})")

    try:
        # Run MegaDetector on all frames (convert Path objects to strings)
        frame_files = [str(p) for p in sorted(frames_dir.glob('*.jpg'))]

        if not frame_files:
            logger.warning("No frames found for MegaDetector processing")
            return {'images': []}

        results = load_and_run_detector_batch(
            model_file='MDV5A',
            image_file_names=frame_files,
            checkpoint_path=None,
            confidence_threshold=detection_threshold,
            checkpoint_frequency=-1,
            quiet=True
        )

        # MegaDetector returns a list, wrap it in the expected format
        if isinstance(results, list):
            results_dict = {'images': results}
        else:
            results_dict = results

        # Save results to JSON
        with open(output_json, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.debug(f"MegaDetector found {len(results_dict.get('images', []))} images with detections")
        return results_dict

    except Exception as e:
        raise RuntimeError(f"MegaDetector failed: {e}") from e


def crop_detections(frames_dir: Path, detections: Dict[str, Any], crops_dir: Path,
                    min_confidence: float = 0.2, category_filter: Optional[List[str]] = None) -> List[Path]:
    """Crop detected animals from frames based on MegaDetector bounding boxes.

    Args:
        frames_dir: Directory containing original frames
        detections: MegaDetector results JSON
        crops_dir: Output directory for cropped images
        min_confidence: Minimum detection confidence (default: 0.2)
        category_filter: List of categories to include (default: ['1'] for animals only)

    Returns:
        List of paths to cropped images
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL (Pillow) is not installed. Install with: pip install Pillow")

    if category_filter is None:
        category_filter = ['1']  # Category 1 = animals in MegaDetector

    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = []

    for img_result in detections.get('images', []):
        img_path = Path(img_result['file'])
        img_detections = img_result.get('detections', [])

        if not img_detections:
            continue

        # Load image
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            continue

        # Crop each detection
        for i, det in enumerate(img_detections):
            confidence = det.get('conf', 0.0)
            category = str(det.get('category', ''))

            # Filter by confidence and category
            if confidence < min_confidence or category not in category_filter:
                continue

            # MegaDetector bbox format: [x_min, y_min, width, height] in normalized coordinates
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue

            x_min, y_min, width, height = bbox

            # Convert to absolute pixel coordinates
            left = int(x_min * img_width)
            top = int(y_min * img_height)
            right = int((x_min + width) * img_width)
            bottom = int((y_min + height) * img_height)

            # Ensure coordinates are within image bounds
            left = max(0, min(left, img_width))
            right = max(0, min(right, img_width))
            top = max(0, min(top, img_height))
            bottom = max(0, min(bottom, img_height))

            # Skip if bbox is invalid
            if right <= left or bottom <= top:
                continue

            # Crop and save
            crop = img.crop((left, top, right, bottom))
            crop_filename = f"{img_path.stem}_crop{i:03d}_conf{int(confidence*100):02d}.jpg"
            crop_path = crops_dir / crop_filename
            crop.save(crop_path, quality=95)
            crop_paths.append(crop_path)

    logger.debug(f"Extracted {len(crop_paths)} animal crops from {len(detections.get('images', []))} frames")
    return crop_paths


def run_speciesnet(image_dir: Path, output_json: Path, country: Optional[str]):
    """Run speciesnet model on extracted frames."""
    cmd = [
        sys.executable,
        '-m', 'speciesnet.scripts.run_model',
        '--folders', str(image_dir),
        '--predictions_json', str(output_json),
    ]
    if country:
        cmd.extend(['--country', country])

    logger.debug(f"Running speciesnet on {image_dir}")
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=SPECIESNET_TIMEOUT,
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        raise RuntimeError("speciesnet module not found. Is it installed?") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"speciesnet timed out after {SPECIESNET_TIMEOUT}s") from None
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else f"exit code {e.returncode}"
        raise RuntimeError(f"speciesnet failed: {error_msg}") from e


def load_predictions(predictions_json: Path) -> Dict[str, Any]:
    """Load predictions from JSON file."""
    if not predictions_json.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_json}")
    try:
        with open(predictions_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {predictions_json}: {e}") from e


def _parse_species_from_prediction(prediction_str: str) -> str:
    """Extract species name from SpeciesNet prediction string.

    Format: "uuid;kingdom;order;family;genus;species;common name"
    Example: "eb3829b0-772e-4088-ae90-f11b9fe38284;mammalia;artiodactyla;cervidae;cervus;elaphus;red deer"
    """
    if not prediction_str or not isinstance(prediction_str, str):
        return 'no_confident_prediction'

    parts = prediction_str.split(';')
    if len(parts) >= 7:
        # Last part is the common name
        common_name = parts[-1].strip()
        if common_name and common_name.lower() not in ('blank', 'animal', 'person', 'vehicle'):
            return common_name

    # Fallback to the last non-empty part
    for part in reversed(parts):
        part = part.strip()
        if part and part.lower() not in ('', 'blank'):
            return part

    return 'no_confident_prediction'


def summarize_predictions(predictions: Dict[str, Any], confidence_threshold: float) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int]]]:
    """Summarize predictions into per-frame results and top species counts."""
    species_counter = Counter()
    per_frame = []

    image_entries = predictions.get('predictions', []) if isinstance(predictions, dict) else []

    for item in image_entries:
        file_name = item.get('filepath', item.get('file', ''))

        # Get the ensemble prediction and score from SpeciesNet
        prediction_str = item.get('prediction', '')
        prediction_score = item.get('prediction_score', 0.0)

        try:
            confidence = float(prediction_score or 0.0)
        except (ValueError, TypeError):
            confidence = 0.0

        # Parse species name from prediction string
        if confidence >= confidence_threshold:
            species_label = _parse_species_from_prediction(prediction_str)
        else:
            species_label = 'no_confident_prediction'

        per_frame.append({
            'frame': file_name,
            'best_label': species_label,
            'confidence': round(confidence, 4),
        })

        if species_label and species_label != 'no_confident_prediction':
            species_counter[species_label] += 1

    top_species = species_counter.most_common(5)
    return per_frame, top_species


def process_video(video_path: Path, done_dir: Path, results_dir: Path, frame_interval: float,
                  confidence_threshold: float, country: Optional[str], frame_quality: int = 2,
                  resize_width: Optional[int] = None, hwaccel: Optional[str] = None,
                  use_megadetector: bool = False, megadetector_threshold: float = 0.2) -> Dict[str, Any]:
    """Process a single video: extract frames, run speciesnet, summarize results.

    Args:
        use_megadetector: If True, use MegaDetector to detect animals before SpeciesNet classification
        megadetector_threshold: Minimum confidence for MegaDetector detections
    """
    logger.info(f"Processing: {video_path.name}{' (with MegaDetector)' if use_megadetector else ''}")
    stem = sanitize_name(video_path.stem)
    video_result_dir = results_dir / stem
    video_result_dir.mkdir(parents=True, exist_ok=True)

    # Create permanent frames directory within results
    frames_dir = video_result_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(video_path, frames_dir, frame_interval, frame_quality, resize_width, hwaccel)
    if not frames:
        raise RuntimeError('No frames were extracted from the video.')

    predictions_json = video_result_dir / f'{stem}_predictions.json'

    # Remove old predictions file to prevent SpeciesNet from trying to resume with mismatched paths
    if predictions_json.exists():
        logger.debug(f"Removing old predictions file: {predictions_json}")
        predictions_json.unlink()

    # Choose workflow: MegaDetector + SpeciesNet OR SpeciesNet only
    if use_megadetector:
        # Step 1: Run MegaDetector to detect animals
        megadetector_json = video_result_dir / f'{stem}_megadetector.json'
        detections = run_megadetector(frames_dir, megadetector_json, megadetector_threshold)

        # Step 2: Crop detected animals
        crops_dir = video_result_dir / 'crops'
        crop_paths = crop_detections(frames_dir, detections, crops_dir,
                                    min_confidence=megadetector_threshold,
                                    category_filter=['1'])  # Only animals

        if not crop_paths:
            logger.warning(f"No animals detected by MegaDetector in {video_path.name}")
            # Return empty results
            per_frame = []
            top_species = []
        else:
            # Step 3: Run SpeciesNet on cropped animals
            logger.debug(f"Running SpeciesNet on {len(crop_paths)} animal crops")
            run_speciesnet(crops_dir, predictions_json, country)
            predictions = load_predictions(predictions_json)
            per_frame, top_species = summarize_predictions(predictions, confidence_threshold)
    else:
        # Original workflow: SpeciesNet on full frames
        run_speciesnet(frames_dir, predictions_json, country)
        predictions = load_predictions(predictions_json)
        per_frame, top_species = summarize_predictions(predictions, confidence_threshold)

    done_dir.mkdir(parents=True, exist_ok=True)
    destination = done_dir / video_path.name
    counter = 0
    while destination.exists():
        counter += 1
        destination = done_dir / f'{video_path.stem}_{counter}{video_path.suffix}'

    shutil.move(video_path, destination)
    logger.info(f"Completed: {video_path.name} -> {len(per_frame)} frames, moved to {destination.name}")

    return {
        'video': video_path.name,
        'frames_processed': len(per_frame),
        'top_species': top_species,
        'predictions_json': str(predictions_json),
        'moved_to': str(destination),
    }


def validate_args(args):
    """Validate command-line arguments."""
    if args.frame_interval <= 0:
        raise ValueError('--frame-interval must be greater than 0')
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError('--confidence-threshold must be between 0 and 1')
    if args.max_workers < 1:
        raise ValueError('--max-workers must be at least 1')
    if not 2 <= args.frame_quality <= 31:
        raise ValueError('--frame-quality must be between 2 (best) and 31 (worst)')
    if args.resize_width is not None and args.resize_width < 100:
        raise ValueError('--resize-width must be at least 100 pixels')
    if not 0 <= args.megadetector_threshold <= 1:
        raise ValueError('--megadetector-threshold must be between 0 and 1')


def process_video_wrapper(args_tuple: Tuple[Path, Path, Path, float, float, Optional[str], int, Optional[int], Optional[str], bool, float]) -> Dict[str, Any]:
    """Wrapper for process_video to enable parallel processing."""
    try:
        return process_video(*args_tuple)
    except Exception as e:
        video_path = args_tuple[0]
        logger.error(f"Failed to process {video_path.name}: {e}")
        logger.debug(traceback.format_exc())
        return {
            'video': video_path.name,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def print_result(result: Dict[str, Any]):
    """Print processing result to console."""
    if 'error' in result:
        logger.error(f"Failed: {result['video']} -> {result['error']}")
        return

    species_text = ', '.join(f'{name} ({count} frames)' for name, count in result['top_species'])
    if not species_text:
        species_text = 'no confident species predictions'

    logger.info(f"Result for {result['video']}: {result['frames_processed']} frames processed")
    logger.info(f"  Top detections: {species_text}")
    logger.info(f"  JSON: {result['predictions_json']}")
    logger.info(f"  Moved to: {result['moved_to']}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch-process wildlife videos with SpeciesNet.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-dir', default='input', help='Folder containing videos to process.')
    parser.add_argument('--done-dir', default='done', help='Folder where processed videos are moved.')
    parser.add_argument('--results-dir', default='results', help='Folder where JSON results are written.')
    parser.add_argument('--frame-interval', type=float, default=2.0, help='Extract one frame every N seconds.')
    parser.add_argument('--confidence-threshold', type=float, default=0.20, help='Minimum detection confidence (0-1).')
    parser.add_argument('--country', default=None, help='Optional ISO-3 country code, e.g. DEU.')
    parser.add_argument('--max-workers', type=int, default=1, help='Number of videos to process in parallel.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging.')

    # Performance optimization arguments
    perf_group = parser.add_argument_group('performance optimization')
    perf_group.add_argument('--frame-quality', type=int, default=2,
                           help='JPEG quality for extracted frames (2=best, 31=worst). Higher values = lower quality but faster.')
    perf_group.add_argument('--resize-width', type=int, default=None,
                           help='Resize frames to this width (maintains aspect ratio). Smaller = faster inference.')
    perf_group.add_argument('--hwaccel', choices=['videotoolbox', 'cuda'], default=None,
                           help='Hardware acceleration for ffmpeg. Use "videotoolbox" on macOS or "cuda" for NVIDIA GPU.')
    perf_group.add_argument('--use-megadetector', action='store_true',
                           help='Use MegaDetector to detect animals before species classification. Improves accuracy but requires additional dependencies.')
    perf_group.add_argument('--megadetector-threshold', type=float, default=0.2,
                           help='Minimum confidence threshold for MegaDetector animal detections (0-1).')
    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f'Argument error: {e}')
        raise SystemExit(1) from e

    if not check_ffmpeg():
        logger.error('ffmpeg is required but was not found in PATH.')
        raise SystemExit(1)

    # Check MegaDetector availability if requested
    if args.use_megadetector:
        if not MEGADETECTOR_AVAILABLE:
            logger.error('MegaDetector is not installed. Install with: pip install megadetector')
            raise SystemExit(1)
        if not PIL_AVAILABLE:
            logger.error('Pillow is not installed. Install with: pip install Pillow')
            raise SystemExit(1)
        logger.info('MegaDetector mode enabled - will detect animals before classification')

    input_dir = Path(args.input_dir)
    done_dir = Path(args.done_dir)
    results_dir = Path(args.results_dir)

    if not input_dir.exists():
        logger.error(f'Input folder does not exist: {input_dir}')
        raise SystemExit(1)

    if not input_dir.is_dir():
        logger.error(f'Input path is not a directory: {input_dir}')
        raise SystemExit(1)

    videos = find_videos(input_dir)
    if not videos:
        logger.warning(f'No videos found in {input_dir}')
        return

    logger.info(f'Found {len(videos)} video(s) to process')

    # Prepare arguments for each video
    video_args = [
        (video_path, done_dir, results_dir, args.frame_interval, args.confidence_threshold,
         args.country, args.frame_quality, args.resize_width, args.hwaccel,
         args.use_megadetector, args.megadetector_threshold)
        for video_path in videos
    ]

    # Process videos (in parallel if max_workers > 1)
    if args.max_workers == 1:
        logger.info('Processing videos sequentially')
        results = [process_video_wrapper(args) for args in video_args]
    else:
        logger.info(f'Processing videos in parallel with {args.max_workers} workers')
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(executor.map(process_video_wrapper, video_args))

    # Print summary
    logger.info('\n' + '=' * 60)
    logger.info('Processing Summary')
    logger.info('=' * 60)

    success_count = sum(1 for r in results if 'error' not in r)
    failed_count = len(results) - success_count

    for result in results:
        print_result(result)

    logger.info('=' * 60)
    logger.info(f'Total: {len(results)} videos | Success: {success_count} | Failed: {failed_count}')
    logger.info('=' * 60)

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
