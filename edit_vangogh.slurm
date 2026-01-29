#!/bin/bash
#SBATCH --job-name=vangogh_edit
#SBATCH --partition=scalar100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vangogh_edit_%j.out
#SBATCH --error=logs/vangogh_edit_%j.err

echo "========================================="
echo "Video Editing: Van Gogh Style"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================="

cd /home/hayk.minasyan/Project/Video-3DGS-new

# Activate environment
source venv_video3dgs_new/bin/activate

# Load CUDA
module load cuda11.7/toolkit/11.7.1
export CUDA_HOME=/cm/shared/apps/cuda11.7/toolkit/11.7.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check GPU
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Dataset (already reconstructed!)
SOURCE_DATA="datasets/edit/DAVIS/480p_frames/blackswan"
VIDEO_NAME="blackswan"
ITERS=3000

echo ""
echo "Dataset: $SOURCE_DATA"
echo "Iterations: $ITERS"
echo "Editor: Text2Video-Zero"
echo "Edit: Black Swan → White Swan (Object Change)"
echo ""

# Run Video Editing with Text2Video-Zero
echo "========================================="
echo "Running Video Editing (White Swan)..."
echo "========================================="

PROMPT="white swan"

python3 main_3dgsvid.py -s $SOURCE_DATA \
    --iteration=$ITERS \
    --use_dual \
    --group_size=1 \
    --deform_type "multi" \
    --random_pts_num 60000 \
    --radius 1 3 \
    --tfloss_weight 2.0 \
    --editing_method="norecursive_single" \
    --initial_editor 1 \
    --prompt "$PROMPT" \
    --cate="obj" \
    --recursive_num 0 \
    --cuda_num=0

if [ $? -ne 0 ]; then
    echo "ERROR: Editing failed!"
    exit 1
fi

echo ""
echo "✅ Video editing completed!"
echo ""

# Find and create edited video
echo "========================================="
echo "Creating edited video..."
echo "========================================="

# Find the edited output directory
EDITED_DIR=$(find output -type d -name "*sty*" -o -name "*vangogh*" -o -name "*edit*" | grep -v "initial_edit" | head -1)

if [ -n "$EDITED_DIR" ]; then
    echo "Found edited directory: $EDITED_DIR"
    
    # Find renders folder
    RENDER_DIR=$(find "$EDITED_DIR" -type d -name "renders" | head -1)
    
    if [ -n "$RENDER_DIR" ] && [ -d "$RENDER_DIR" ]; then
        echo "Creating video from: $RENDER_DIR"
        
        ffmpeg -framerate 10 -pattern_type glob -i "$RENDER_DIR/*.png" \
               -c:v libx264 -pix_fmt yuv420p -crf 18 -y \
               "output/blackswan_VANGOGH_STYLE.mp4" 2>&1 | tail -10
        
        if [ -f "output/blackswan_VANGOGH_STYLE.mp4" ]; then
            echo ""
            echo "✅✅✅ SUCCESS! Edited video created!"
            echo ""
            ls -lh "output/blackswan_VANGOGH_STYLE.mp4"
        fi
    fi
else
    echo "Searching for any output folders..."
    find output -type d -name "renders" | head -5
fi

echo ""
echo "========================================="
echo "🎨 Van Gogh Style Editing Complete! 🎨"
echo "Finished at: $(date)"
echo "========================================="
echo ""
echo "Results:"
find output -name "*VANGOGH*.mp4" -o -name "*edit*.mp4" 2>/dev/null
echo ""
echo "To download:"
echo "  scp <server>:/home/hayk.minasyan/Project/Video-3DGS-new/output/blackswan_VANGOGH_STYLE.mp4 ."
echo ""

