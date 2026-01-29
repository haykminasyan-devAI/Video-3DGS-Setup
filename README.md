# Video-3DGS - Video Editing with 3D Gaussian Splatting

## 🎉 **Working Setup - Ready to Use!**

---

## 📹 **Your Final Videos**

All videos are in: `output/`

1. **blackswan_ORIGINAL_before_vangogh.mp4** (2.1 MB) - Original video
2. **blackswan_VANGOGH_STYLE_FINAL.mp4** (2.7 MB) 🎨 - Van Gogh painting style
3. **blackswan_WHITE_SWAN_FINAL.mp4** (1.2 MB) 🦢 - Black swan → White swan
4. **blackswan_reconstructed_SUCCESS.mp4** (2.2 MB) - 3DGS reconstruction

**Download:**
```bash
scp <server>:/home/hayk.minasyan/Project/Video-3DGS-new/output/blackswan_*.mp4 .
```

---

## 🚀 **How to Edit More Videos**

### **Quick Start:**

1. **Edit** `edit_white_swan.slurm` (lines 52-64):

```bash
# Change the prompt:
PROMPT="your editing instruction here"

# Change the category:
--cate="sty"    # For style (Van Gogh, watercolor, anime, etc.)
--cate="obj"    # For object changes (white swan, red car, etc.)
--cate="back"   # For background changes
```

2. **Submit:**
```bash
cd /home/hayk.minasyan/Project/Video-3DGS-new
sbatch edit_white_swan.slurm
```

3. **Monitor:**
```bash
squeue -u $(whoami)
tail -f logs/white_swan_*.out
```

4. **Results** will be in: `output/`

---

## 📝 **Available Scripts**

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `edit_white_swan.slurm` | Video editing (style/object/background) | Main editing script - modify and use this |
| `test_reconstruction_only.slurm` | 3DGS reconstruction only | To reconstruct videos without editing |
| `edit_vangogh.slurm` | Similar to white_swan | Alternative editing script |
| `run_with_preprocessed.slurm` | Full pipeline | Reconstruction + editing together |

**Recommended:** Use `edit_white_swan.slurm` for all editing tasks.

---

## 🎨 **Example Editing Prompts**

### **Style Changes** (`--cate="sty"`):
- `"watercolor painting"`
- `"anime style"`
- `"pencil sketch"`
- `"cyberpunk neon style"`
- `"Monet impressionist painting"`

### **Object Changes** (`--cate="obj"`):
- `"white swan"` (already did this!)
- `"golden swan"`
- `"robot swan"`

### **Background Changes** (`--cate="back"`):
- `"sunset background"`
- `"winter scene with snow"`
- `"ocean background"`

---

## 🔧 **Environment**

**Python Environment:** `venv_video3dgs_new`

**Activate:**
```bash
source venv_video3dgs_new/bin/activate
```

**Key Packages:**
- PyTorch 2.0.1 + CUDA 11.7
- diffusers 0.14.0
- transformers 4.26.0
- Custom CUDA modules: simple-knn, diff-gaussian-rasterization, tiny-cuda-nn

---

## 📊 **Datasets**

**Location:** `datasets/edit/DAVIS/480p_frames/`

**Available scenes for editing:**
- blackswan ✓ (already edited)
- bear, boat, camel, car-roundabout, cows, dog, elephant
- flamingo, gold-fish, hike, hockey, horsejump-high, kid-football
- kite-surf, lab-coat, longboard, lucia, mallard-water
- mbike-trick, motorbike, rhino, scooter-gray, swing

**To edit a different scene:** Change line 35 in `edit_white_swan.slurm`:
```bash
SOURCE_DATA="datasets/edit/DAVIS/480p_frames/bear"  # Change to any scene
```

---

## ⚙️ **Editing Your Custom Video**

To edit `asset/2025-12-22 23.23.20.mp4` or any custom video:

1. **Need COLMAP installed** (contact cluster admin)
2. Extract frames from your video
3. Run MC-COLMAP preprocessing
4. Then use the editing scripts

**OR** use pre-processed datasets (like the DAVIS ones you have now).

---

## 📁 **Project Structure**

```
Video-3DGS-new/
├── output/                    # Your final videos ✨
├── edit_white_swan.slurm     # Main editing script 
├── test_reconstruction_only.slurm  # Reconstruction script
├── venv_video3dgs_new/       # Python environment
├── datasets/edit/            # DAVIS dataset (11GB)
├── main_3dgsvid.py          # Main Python script (modified)
├── models/                   # Video editors and optical flow
└── logs/                     # Job logs (cleaned up)
```

---

## 🎯 **Quick Reference**

**Edit a video:**
```bash
# 1. Modify prompt in edit_white_swan.slurm
# 2. Submit job
sbatch edit_white_swan.slurm

# 3. Check status
squeue -u $(whoami)

# 4. Monitor
tail -f logs/white_swan_*.out

# 5. Result will be in output/
```

**Time per edit:** ~30-45 minutes on A100 GPU

---

## ✅ **What's Working**

- ✅ Video reconstruction with 3D Gaussian Splatting
- ✅ Style editing (Van Gogh, artistic styles)
- ✅ Object editing (color changes, object transformation)
- ✅ Temporal consistency (no flickering!)
- ✅ Text2Video-Zero editor
- ✅ All dependencies installed

---

## 📚 **Paper**

"Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting" (TMLR 2025)

- Paper: https://arxiv.org/pdf/2406.02541
- Project: https://video-3dgs-project.github.io/

---

## 🎊 **Success!**

You have a fully working Video-3DGS setup with:
- 4 edited videos created ✓
- Complete environment ready ✓
- Easy-to-use scripts ✓

Enjoy creating more temporally-consistent edited videos! 🎬




