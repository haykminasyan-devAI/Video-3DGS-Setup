# Video-3DGS

Implementation of "Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting" (TMLR 2025).

This repository provides tools for video editing using 3D Gaussian Splatting reconstruction to achieve temporal consistency.

## Overview

Video-3DGS addresses temporal inconsistency in video editing by:
1. Reconstructing videos using 3D Gaussian Splatting
2. Applying text-guided editing with diffusion models
3. Maintaining temporal coherence across frames

## Features

- Video reconstruction with 3D Gaussian Splatting
- Style transfer (artistic styles, painting effects)
- Object editing (color, shape transformations)
- Background modification
- Temporal consistency preservation

## Installation

### Requirements

- Python 3.8+
- CUDA 11.7
- PyTorch 2.0.1

### Setup

```bash
# Activate virtual environment
source venv_video3dgs_new/bin/activate

# Key dependencies
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install diffusers==0.14.0 transformers==4.26.0
```

### Custom CUDA Modules

The following modules are built from source:
- simple-knn
- diff-gaussian-rasterization  
- tiny-cuda-nn

## Usage

### Basic Video Editing

Edit the configuration in `edit_white_swan.slurm`:

```bash
PROMPT="your editing instruction"
--cate="sty"    # style, obj (object), or back (background)
```

Submit the job:

```bash
sbatch edit_white_swan.slurm
```

Monitor progress:

```bash
squeue -u $(whoami)
tail -f logs/white_swan_*.out
```

Results are saved to `output/`.

### Editing Categories

**Style Transfer** (`--cate="sty"`):
- Watercolor painting
- Anime style
- Van Gogh style
- Pencil sketch

**Object Editing** (`--cate="obj"`):
- Color changes
- Object transformation

**Background Editing** (`--cate="back"`):
- Scene modification
- Environment changes

### Available Scripts

| Script | Description |
|--------|-------------|
| `edit_white_swan.slurm` | Main video editing pipeline |
| `test_reconstruction_only.slurm` | 3DGS reconstruction only |
| `edit_vangogh.slurm` | Alternative editing script |
| `run_with_preprocessed.slurm` | Full reconstruction + editing |

### Processing Time

Approximate runtime on A100 GPU: 30-45 minutes per video

## Dataset

Pre-processed DAVIS dataset scenes available in `datasets/edit/DAVIS/480p_frames/`:

- blackswan, bear, boat, camel, car-roundabout, cows, dog, elephant
- flamingo, gold-fish, hike, hockey, horsejump-high, kid-football  
- kite-surf, lab-coat, longboard, lucia, mallard-water
- mbike-trick, motorbike, rhino, scooter-gray, swing

To use a different scene, modify the `SOURCE_DATA` path in the SLURM script.

## Custom Video Processing

For custom videos:

1. Install COLMAP for structure-from-motion
2. Extract video frames
3. Run MC-COLMAP preprocessing
4. Execute editing scripts

Alternatively, use the provided pre-processed DAVIS dataset.

## Project Structure

```
Video-3DGS-new/
├── output/                   # Generated videos
├── datasets/edit/            # Input datasets
├── main_3dgsvid.py          # Main implementation
├── models/                   # Video editors and optical flow
│   ├── video_editors/
│   └── optical_flow/
├── scene/                    # 3DGS scene representation
├── gaussian_renderer/        # Gaussian splatting renderer
├── utils/                    # Utility functions
└── *.slurm                   # Job submission scripts
```

## Citation

```bibtex
@article{video3dgs2025,
  title={Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

## References

- Paper: https://arxiv.org/pdf/2406.02541
- Project Page: https://video-3dgs-project.github.io/

## License

See LICENSE file for details.




