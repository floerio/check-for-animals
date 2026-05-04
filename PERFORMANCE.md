# Performance Optimization Guide

This guide explains how to speed up video processing using the available optimization flags.

## Quick Start: Recommended Fast Settings

**Best Accuracy** (with MegaDetector - recommended for wildlife monitoring):
```bash
python3 check-for-animals.py \
  --use-megadetector \
  --frame-interval 5.0 \
  --resize-width 640 \
  --hwaccel videotoolbox \
  --max-workers 2
```

**Best Speed** (macOS, without MegaDetector):
```bash
python3 check-for-animals.py \
  --frame-interval 5.0 \
  --resize-width 640 \
  --hwaccel videotoolbox \
  --frame-quality 5 \
  --max-workers 2
```

**Linux/Windows with NVIDIA GPU**:
```bash
python3 check-for-animals.py \
  --use-megadetector \
  --frame-interval 5.0 \
  --resize-width 640 \
  --hwaccel cuda \
  --max-workers 2
```

**Note:** SpeciesNet automatically uses GPU acceleration (M1/M2 Mac via MPS, NVIDIA via CUDA) if PyTorch detects compatible hardware. No additional flags needed.

## MegaDetector Integration (Recommended for Better Accuracy)

**MegaDetector** is Microsoft's wildlife detection model that can significantly improve detection accuracy. When enabled, the workflow changes from:

```
Video → Frames → SpeciesNet → Species predictions
```

to:

```
Video → Frames → MegaDetector (detect animals) → Crop animals → SpeciesNet (classify) → Species predictions
```

### When to Use MegaDetector:

✅ **Use MegaDetector if you experience:**
- Animals present but not detected
- Wrong species classifications
- Need for higher accuracy in challenging conditions (camouflage, distant animals, low light)

❌ **Skip MegaDetector if:**
- Processing speed is critical and accuracy is acceptable
- You have simple, clear camera trap images with obvious animals

### Enabling MegaDetector:

First, install the additional dependencies:
```bash
pip install megadetector Pillow
```

Then add the `--use-megadetector` flag:
```bash
python3 check-for-animals.py \
  --use-megadetector \
  --frame-interval 5.0 \
  --resize-width 640 \
  --max-workers 2
```

### MegaDetector Options:

- `--use-megadetector`: Enable MegaDetector workflow
- `--megadetector-threshold`: Minimum confidence for detections (default: 0.2, range: 0-1)

**Example with custom threshold:**
```bash
python3 check-for-animals.py \
  --use-megadetector \
  --megadetector-threshold 0.3 \  # More conservative, fewer false positives
  --frame-interval 5.0
```

**Trade-offs:**
- ⚡ **Speed**: ~20-30% slower (running two models instead of one)
- ✅ **Accuracy**: Significantly better detection and classification
- 📊 **Best for**: Camera trap workflows, wildlife monitoring

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

### 4. GPU Acceleration (Automatic - Huge Impact if available)

SpeciesNet **automatically** uses GPU acceleration when available:
- **M1/M2/M3 Mac:** Uses MPS (Metal Performance Shaders) automatically
- **NVIDIA GPU:** Uses CUDA automatically if drivers are installed
- No manual flags required - just ensure PyTorch is installed correctly

**Performance impact:** 10-100x faster inference on compatible hardware.

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

## Verifying GPU Acceleration

### For M1/M2/M3 Mac:
Check if MPS (Metal Performance Shaders) is available:
```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

If it shows `True`, SpeciesNet will automatically use your Mac's GPU. No additional setup needed!

### For NVIDIA GPU (Linux/Windows):
Check if CUDA is available:
```bash
nvidia-smi  # Should show GPU info
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is not available but you have an NVIDIA GPU, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
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
--frame-interval 10.0 --resize-width 480 --hwaccel videotoolbox --frame-quality 8 --max-workers 4
```

## Troubleshooting

**"hwaccel not supported"**: Your ffmpeg may not have hardware acceleration compiled in. Remove `--hwaccel`.

**Slow inference on M1/M2 Mac**: Check if MPS is available (see "Verifying GPU Acceleration" above). PyTorch should use your Mac's GPU automatically.

**System slowdown**: Reduce `--max-workers` to prevent overloading your system.

**Low accuracy**: Increase `--resize-width` or decrease `--frame-interval` to process more/better frames.
