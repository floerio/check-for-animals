#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.mts', '.m4v'}


def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def sanitize_name(name: str) -> str:
    if not name:
        return 'video'
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
    return safe.strip('_') or 'video'


def find_videos(input_dir: Path):
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])


def extract_frames(video_path: Path, frames_dir: Path, every_n_seconds: float):
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / 'frame_%06d.jpg')
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(video_path),
        '-vf', f'fps=1/{every_n_seconds}',
        '-q:v', '2',
        pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(frames_dir.glob('*.jpg'))


def run_speciesnet(image_dir: Path, output_json: Path, country: Optional[str]):
    cmd = [
        sys.executable,
        '-m', 'speciesnet.scripts.run_model',
        '--folders', str(image_dir),
        '--predictions_json', str(output_json),
    ]
    if country:
        cmd.extend(['--country', country])
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError("speciesnet module not found. Is it installed?") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"speciesnet failed with exit code {e.returncode}") from e


def load_predictions(predictions_json: Path):
    if not predictions_json.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_json}")
    with open(predictions_json, 'r', encoding='utf-8') as f:
        return json.load(f)


def summarize_predictions(predictions, confidence_threshold: float):
    species_counter = Counter()
    per_frame = []

    image_entries = predictions.get('predictions', []) if isinstance(predictions, dict) else []

    for item in image_entries:
        file_name = item.get('filepath', item.get('file', ''))
        detections = item.get('detections', [])
        best_label = None
        best_conf = 0.0

        for det in detections:
            conf = float(det.get('conf', 0.0) or 0.0)
            if conf < confidence_threshold:
                continue

            label = None

            classifications = det.get('classifications') or det.get('classification') or []
            if isinstance(classifications, list) and classifications:
                first = classifications[0]
                if isinstance(first, list) and len(first) >= 2:
                    label = str(first[0])
                    class_conf = float(first[1] or 0.0)
                    if class_conf > best_conf:
                        best_conf = class_conf
                        best_label = label
                elif isinstance(first, dict):
                    candidate = first.get('label') or first.get('class_name') or first.get('name')
                    class_conf = float(first.get('conf', first.get('score', 0.0)) or 0.0)
                    if candidate and class_conf > best_conf:
                        best_conf = class_conf
                        best_label = str(candidate)

            if best_label is None:
                category = det.get('category')
                if category == '1' or category == 1:
                    best_label = 'animal_unspecified'
                    best_conf = conf

        per_frame.append({
            'frame': file_name,
            'best_label': best_label or 'no_confident_prediction',
            'confidence': round(best_conf, 4),
        })

        if best_label and best_label != 'no_confident_prediction':
            species_counter[best_label] += 1

    top_species = species_counter.most_common(5)
    return per_frame, top_species


def write_csv(csv_path: Path, rows):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'best_label', 'confidence'])
        writer.writeheader()
        writer.writerows(rows)


def process_video(video_path: Path, done_dir: Path, results_dir: Path, frame_interval: float, confidence_threshold: float, country: Optional[str]):
    stem = sanitize_name(video_path.stem)
    video_result_dir = results_dir / stem
    video_result_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f'{stem}_frames_') as tmpdir:
        frames_dir = Path(tmpdir)
        frames = extract_frames(video_path, frames_dir, frame_interval)
        if not frames:
            raise RuntimeError('No frames were extracted from the video.')

        predictions_json = video_result_dir / f'{stem}_predictions.json'
        run_speciesnet(frames_dir, predictions_json, country)
        predictions = load_predictions(predictions_json)
        per_frame, top_species = summarize_predictions(predictions, confidence_threshold)
        csv_path = video_result_dir / f'{stem}_summary.csv'
        write_csv(csv_path, per_frame)

    done_dir.mkdir(parents=True, exist_ok=True)
    destination = done_dir / video_path.name
    # Use a unique suffix to avoid race conditions
    counter = 0
    while destination.exists():
        counter += 1
        destination = done_dir / f'{video_path.stem}_{counter}{video_path.suffix}'
    shutil.move(video_path, destination)

    return {
        'video': video_path.name,
        'frames_processed': len(per_frame),
        'top_species': top_species,
        'predictions_json': str(predictions_json),
        'summary_csv': str(csv_path),
        'moved_to': str(destination),
    }


def validate_args(args):
    if args.frame_interval <= 0:
        raise ValueError('--frame-interval must be greater than 0')
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError('--confidence-threshold must be between 0 and 1')


def main():
    parser = argparse.ArgumentParser(description='Batch-process wildlife videos with SpeciesNet.')
    parser.add_argument('--input-dir', default='input', help='Folder containing videos to process.')
    parser.add_argument('--done-dir', default='done', help='Folder where processed videos are moved.')
    parser.add_argument('--results-dir', default='results', help='Folder where JSON/CSV results are written.')
    parser.add_argument('--frame-interval', type=float, default=2.0, help='Extract one frame every N seconds.')
    parser.add_argument('--confidence-threshold', type=float, default=0.20, help='Minimum detection confidence.')
    parser.add_argument('--country', default=None, help='Optional ISO-3 country code, e.g. DEU.')
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        raise SystemExit(f'Argument error: {e}') from e

    if not check_ffmpeg():
        raise SystemExit('ffmpeg is required but was not found in PATH.')

    input_dir = Path(args.input_dir)
    done_dir = Path(args.done_dir)
    results_dir = Path(args.results_dir)

    if not input_dir.exists():
        raise SystemExit(f'Input folder does not exist: {input_dir}')

    videos = find_videos(input_dir)
    if not videos:
        print(f'No videos found in {input_dir}', file=sys.stderr)
        return

    for video_path in videos:
        print(f'\nProcessing: {video_path.name}')
        try:
            result = process_video(
                video_path=video_path,
                done_dir=done_dir,
                results_dir=results_dir,
                frame_interval=args.frame_interval,
                confidence_threshold=args.confidence_threshold,
                country=args.country,
            )
            species_text = ', '.join(f'{name} ({count} frames)' for name, count in result['top_species'])
            if not species_text:
                species_text = 'no confident species predictions'
            print(f"Result for {result['video']}: {result['frames_processed']} frames processed; top detections: {species_text}")
            print(f"JSON: {result['predictions_json']}")
            print(f"CSV:  {result['summary_csv']}")
            print(f"Moved to: {result['moved_to']}")
        except Exception as e:
            print(f'Failed: {video_path.name} -> {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
