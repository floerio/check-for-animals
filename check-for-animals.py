#!/usr/bin/env python3
import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

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


def run_speciesnet(image_dir: Path, output_json: Path, country: Optional[str], use_gpu: bool = False):
    """Run speciesnet model on extracted frames."""
    cmd = [
        sys.executable,
        '-m', 'speciesnet.scripts.run_model',
        '--folders', str(image_dir),
        '--predictions_json', str(output_json),
    ]
    if country:
        cmd.extend(['--country', country])
    if use_gpu:
        cmd.append('--use_gpu')

    logger.debug(f"Running speciesnet on {image_dir}{' (GPU)' if use_gpu else ' (CPU)'}")
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


def _extract_classification_label(classifications: Any) -> Optional[Tuple[str, float]]:
    """Extract label and confidence from classification data."""
    if not isinstance(classifications, list) or not classifications:
        return None

    first = classifications[0]

    # Handle list format: [label, confidence]
    if isinstance(first, list) and len(first) >= 2:
        try:
            return str(first[0]), float(first[1] or 0.0)
        except (ValueError, TypeError):
            return None

    # Handle dict format
    if isinstance(first, dict):
        label = first.get('label') or first.get('class_name') or first.get('name')
        conf = first.get('conf', first.get('score', 0.0))
        if label:
            try:
                return str(label), float(conf or 0.0)
            except (ValueError, TypeError):
                return None

    return None


def _get_best_detection(detections: List[Dict[str, Any]], confidence_threshold: float) -> Tuple[Optional[str], float]:
    """Find the best species detection from a list of detections."""
    best_label = None
    best_conf = 0.0

    for det in detections:
        try:
            conf = float(det.get('conf', 0.0) or 0.0)
        except (ValueError, TypeError):
            continue

        if conf < confidence_threshold:
            continue

        # Try to extract classification
        classifications = det.get('classifications') or det.get('classification') or []
        result = _extract_classification_label(classifications)

        if result:
            label, class_conf = result
            if class_conf > best_conf:
                best_conf = class_conf
                best_label = label
        # Fallback to generic animal detection
        elif best_label is None:
            category = det.get('category')
            if category in ('1', 1):
                best_label = 'animal_unspecified'
                best_conf = conf

    return best_label, best_conf


def summarize_predictions(predictions: Dict[str, Any], confidence_threshold: float) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int]]]:
    """Summarize predictions into per-frame results and top species counts."""
    species_counter = Counter()
    per_frame = []

    image_entries = predictions.get('predictions', []) if isinstance(predictions, dict) else []

    for item in image_entries:
        file_name = item.get('filepath', item.get('file', ''))
        detections = item.get('detections', [])

        best_label, best_conf = _get_best_detection(detections, confidence_threshold)

        per_frame.append({
            'frame': file_name,
            'best_label': best_label or 'no_confident_prediction',
            'confidence': round(best_conf, 4),
        })

        if best_label and best_label != 'no_confident_prediction':
            species_counter[best_label] += 1

    top_species = species_counter.most_common(5)
    return per_frame, top_species


def process_video(video_path: Path, done_dir: Path, results_dir: Path, frame_interval: float,
                  confidence_threshold: float, country: Optional[str], frame_quality: int = 2,
                  resize_width: Optional[int] = None, hwaccel: Optional[str] = None,
                  use_gpu: bool = False) -> Dict[str, Any]:
    """Process a single video: extract frames, run speciesnet, summarize results."""
    logger.info(f"Processing: {video_path.name}")
    stem = sanitize_name(video_path.stem)
    video_result_dir = results_dir / stem
    video_result_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f'{stem}_frames_') as tmpdir:
        frames_dir = Path(tmpdir)
        frames = extract_frames(video_path, frames_dir, frame_interval, frame_quality, resize_width, hwaccel)
        if not frames:
            raise RuntimeError('No frames were extracted from the video.')

        predictions_json = video_result_dir / f'{stem}_predictions.json'

        # Remove old predictions file to prevent SpeciesNet from trying to resume with mismatched paths
        if predictions_json.exists():
            logger.debug(f"Removing old predictions file: {predictions_json}")
            predictions_json.unlink()

        run_speciesnet(frames_dir, predictions_json, country, use_gpu)
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


def process_video_wrapper(args_tuple: Tuple[Path, Path, Path, float, float, Optional[str], int, Optional[int], Optional[str], bool]) -> Dict[str, Any]:
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
    perf_group.add_argument('--gpu', action='store_true',
                           help='Use GPU for SpeciesNet inference (requires CUDA-compatible GPU and proper setup).')
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
         args.country, args.frame_quality, args.resize_width, args.hwaccel, args.gpu)
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
