# Performance Optimization Guide

This guide explains how to speed up video processing using the available optimization flags.

## Quick Start: Recommended Fast Settings

For **macOS** (best balance of speed and accuracy):
```bash
python3 check-for-animals.py \
  --frame-interval 5.0 \
  --resize-width 640 \
  --hwaccel videotoolbox \
  --frame-quality 5 \
  --max-workers 2
```

For **Linux/Windows with NVIDIA GPU** (fastest):
```bash
python3 check-for-animals.py \
  --frame-interval 5.0 \
  --resize-width 640 \
  --hwaccel cuda \
  --gpu \
  --frame-quality 5 \
  --max-workers 2
```

## Optimization Options Explained

### 1. Frame Interval (Biggest Impact)
**Flag:** `--frame-interval <seconds>`  
**Default:** 2.0 seconds

Extract fewer frames from videos:
- `--frame-interval 5.0` → 60% fewer frames to process
- `--frame-interval 10.0` → 80% fewer frames to process

**Tradeoff:** May miss brief animal appearances, but usually sufficient for wildlife monitoring.

### 2. Frame Resolution (Major Impact)
**Flag:** `--resize-width <pixels>`  
**Default:** None (full resolution)

Resize frames before SpeciesNet inference:
- `--resize-width 640` → Good for most wildlife (recommended)
- `--resize-width 480` → Faster, slightly less accurate
- `--resize-width 1280` → Higher accuracy, slower

**Tradeoff:** Smaller frames = faster inference but may miss small/distant animals.

### 3. Hardware Acceleration for ffmpeg (Moderate Impact)
**Flag:** `--hwaccel <type>`  
**Options:** `videotoolbox` (macOS), `cuda` (NVIDIA GPU)

Accelerate frame extraction from videos:
- **macOS:** `--hwaccel videotoolbox` (use Mac's GPU)
- **NVIDIA GPU:** `--hwaccel cuda` (requires CUDA drivers)

**Tradeoff:** Minimal, usually just faster with no quality loss.

### 4. GPU for SpeciesNet (Huge Impact if available)
**Flag:** `--gpu`  
**Default:** CPU-only

Use GPU for species detection (10-100x faster inference):
- Requires NVIDIA GPU with CUDA support
- Requires proper PyTorch/CUDA installation

**Tradeoff:** None if you have compatible hardware.

### 5. Frame Quality (Minor Impact)
**Flag:** `--frame-quality <2-31>`  
**Default:** 2 (highest quality)

Adjust JPEG compression:
- `--frame-quality 2` → Best quality, slower I/O
- `--frame-quality 5` → Good quality, faster (recommended)
- `--frame-quality 10` → Lower quality, fastest

**Tradeoff:** Higher values = faster but may reduce detection accuracy.

### 6. Parallel Video Processing (Linear Speedup)
**Flag:** `--max-workers <number>`  
**Default:** 1 (sequential)

Process multiple videos simultaneously:
- `--max-workers 2` → 2x faster for multiple videos
- `--max-workers 4` → 4x faster (if you have CPU cores)

**Tradeoff:** Uses more CPU/RAM. Don't exceed your CPU core count.

## Performance Comparison

Example processing times for a 10-minute 1080p video:

| Configuration | Frames | Time | Speedup |
|--------------|--------|------|---------|
| **Default** | 300 | ~15 min | 1x |
| **+ interval 5s** | 120 | ~6 min | 2.5x |
| **+ resize 640px** | 120 | ~3 min | 5x |
| **+ hwaccel** | 120 | ~2.5 min | 6x |
| **+ GPU** | 120 | ~1 min | 15x |
| **+ quality 5** | 120 | ~50 sec | 18x |

*Times are approximate and depend on hardware*

## GPU Setup (Optional but Recommended)

### Check if you have NVIDIA GPU:
```bash
nvidia-smi  # Should show GPU info if available
```

### Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU is available:
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Monitoring Performance

Use verbose mode to see timing details:
```bash
python3 check-for-animals.py --verbose ...
```

## Finding the Right Balance

Start conservative and increase aggressiveness:

**Level 1 - Safe (2x faster):**
```bash
--frame-interval 5.0 --max-workers 2
```

**Level 2 - Balanced (5-6x faster):**
```bash
--frame-interval 5.0 --resize-width 640 --hwaccel videotoolbox --max-workers 2
```

**Level 3 - Aggressive (10-20x faster):**
```bash
--frame-interval 10.0 --resize-width 480 --hwaccel videotoolbox --frame-quality 8 --gpu --max-workers 4
```

## Troubleshooting

**"hwaccel not supported"**: Your ffmpeg may not have hardware acceleration compiled in. Remove `--hwaccel`.

**"CUDA not available"**: GPU flag won't work. Install CUDA-enabled PyTorch or remove `--gpu`.

**System slowdown**: Reduce `--max-workers` to prevent overloading your system.

**Low accuracy**: Increase `--resize-width` or decrease `--frame-interval` to process more/better frames.
