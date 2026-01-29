#!/bin/bash
#SBATCH --job-name=white_swan
#SBATCH --partition=scalar100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/white_swan_%j.out
#SBATCH --error=logs/white_swan_%j.err

echo "========================================="
echo "Video Editing: Motorbike in Desert"
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

# Dataset - Motorbike
SOURCE_DATA="datasets/edit/DAVIS/480p_frames/motorbike"
VIDEO_NAME="motorbike"
ITERS=3000

echo ""
echo "Dataset: $SOURCE_DATA"
echo "Iterations: $ITERS"
echo "Editor: Text2Video-Zero"
echo "Edit: Motorbike in Desert (Background Change)"
echo ""

# Run Video Editing - Background Change
echo "========================================="
echo "Changing Background to Desert..."
echo "========================================="

# Background change prompt
PROMPT="desert background"

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
    --cate="back" \
    --recursive_num 0 \
    --cuda_num=0

if [ $? -ne 0 ]; then
    echo "ERROR: Editing failed!"
    exit 1
fi

echo ""
echo "✅ Desert background editing completed!"
echo ""

# Find and create edited video
echo "========================================="
echo "Creating white swan video..."
echo "========================================="

# Find the edited output directory
EDITED_DIR=$(find output/initial_edit -type d -name "*white_swan*" -o -name "*obj*" | head -1)

if [ -n "$EDITED_DIR" ]; then
    echo "Found edited directory: $EDITED_DIR"
    
    # Find vid_output folder
    VID_OUTPUT=$(find "$EDITED_DIR" -type d -name "vid_output_0" | head -1)
    
    if [ -n "$VID_OUTPUT" ] && [ -d "$VID_OUTPUT" ]; then
        echo "Creating video from: $VID_OUTPUT"
        
        ffmpeg -framerate 10 -i "$VID_OUTPUT/%05d.png" \
               -c:v libx264 -pix_fmt yuv420p -crf 18 -y \
               "output/motorbike_DESERT.mp4" 2>&1 | tail -10
        
        if [ -f "output/motorbike_DESERT.mp4" ]; then
            echo ""
            echo "✅✅✅ SUCCESS! Motorbike desert video created!"
            echo ""
            ls -lh "output/motorbike_DESERT.mp4"
        fi
    fi
else
    echo "Searching for output folders..."
    find output -type d -name "*obj*" -o -name "*vid_output*" | head -10
fi

echo ""
echo "========================================="
echo "🏍️ Motorbike in Desert Complete! 🏜️"
echo "Finished at: $(date)"
echo "========================================="
echo ""
echo "Results:"
find output -name "*motorbike*.mp4" -o -name "*DESERT*.mp4" 2>/dev/null
echo ""
echo "To download:"
echo "  scp <server>:/home/hayk.minasyan/Project/Video-3DGS-new/output/motorbike_DESERT.mp4 ."
echo ""


