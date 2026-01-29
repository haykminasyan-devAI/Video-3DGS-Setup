#!/bin/bash
#SBATCH --job-name=v3dgs_recon
#SBATCH --partition=scalar100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/recon_test_%j.out
#SBATCH --error=logs/recon_test_%j.err

echo "========================================="
echo "Video-3DGS Reconstruction Test (No Editing)"
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Dataset with COLMAP preprocessing - use BASE path (code adds _pts_camera_from_deva automatically!)
SOURCE_DATA="datasets/recon/DAVIS/JPEGImages/480p/blackswan"
VIDEO_NAME="blackswan"
ITERS=3000

echo ""
echo "Dataset: $SOURCE_DATA"
echo "Iterations: $ITERS"
echo ""

# Test just reconstruction (Stage 1) - no editing
echo "Running Video Reconstruction (Stage 1 only)..."
echo "========================================="

python3 main_3dgsvid.py -s $SOURCE_DATA \
    --iteration=$ITERS \
    --use_dual \
    --group_size=1 \
    --deform_type "multi" \
    --random_pts_num 60000 \
    --radius 1 3 \
    --tfloss_weight 2.0 \
    --initial_editor 0

if [ $? -ne 0 ]; then
    echo "ERROR: Reconstruction failed!"
    exit 1
fi

echo ""
echo "✅ Reconstruction completed!"
echo "========================================="

# Create video from renders
OUTPUT_DIR="output/${VIDEO_NAME}"

if [ -d "$OUTPUT_DIR/test/ours_${ITERS}/renders" ]; then
    echo "Creating video from rendered frames..."
    ffmpeg -framerate 10 -i "$OUTPUT_DIR/test/ours_${ITERS}/renders/%05d.png" \
           -c:v libx264 -pix_fmt yuv420p -crf 18 -y \
           "output/${VIDEO_NAME}_reconstructed.mp4" 2>&1 | tail -5
    
    if [ -f "output/${VIDEO_NAME}_reconstructed.mp4" ]; then
        echo "✅ Video created: output/${VIDEO_NAME}_reconstructed.mp4"
        ls -lh "output/${VIDEO_NAME}_reconstructed.mp4"
    fi
fi

echo ""
echo "========================================="
echo "🎉 Reconstruction Test Complete! 🎉"
echo "Finished at: $(date)"
echo "========================================="
echo ""
echo "Results:"
find output -name "*.mp4" -o -name "*.png" -path "*/renders/*" | head -10
echo ""
echo "To view reconstruction quality, check:"
echo "  output/${VIDEO_NAME}/test/ours_${ITERS}/renders/"
echo ""

