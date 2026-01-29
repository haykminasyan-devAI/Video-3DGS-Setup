#!/bin/bash
#SBATCH --job-name=v3dgs_test
#SBATCH --partition=scalar100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v3dgs_test_%j.out
#SBATCH --error=logs/v3dgs_test_%j.err

echo "========================================="
echo "Video-3DGS with Pre-Processed Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================="

cd /home/hayk.minasyan/Project/Video-3DGS-new
mkdir -p logs

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

echo ""
echo "========================================="
echo "Testing with Pre-Processed Dataset"
echo "========================================="

# Use blackswan from downloaded DAVIS dataset
# This data already has MC-COLMAP preprocessing done!
SOURCE_DATA="datasets/DAVIS/JPEGImages/480p/blackswan_pts_camera_from_deva"
VIDEO_NAME="blackswan"
ITERS=1500
GNUM=1

if [ ! -d "$SOURCE_DATA" ]; then
    echo "ERROR: Dataset not found at $SOURCE_DATA"
    echo ""
    echo "Please ensure the dataset was extracted properly."
    echo ""
    exit 1
fi

echo "Using dataset: $SOURCE_DATA"
echo ""

# Step 1: Video Reconstruction (no COLMAP needed - already preprocessed!)
echo "Step 1: Running Video Reconstruction..."
echo "========================================="
python3 main_3dgsvid.py -s $SOURCE_DATA \
    --iteration=$ITERS \
    --use_dual \
    --group_size=$GNUM \
    --deform_type "multi" \
    --random_pts_num 60000 \
    --radius 1 3 \
    --tfloss_weight 2.0

if [ $? -ne 0 ]; then
    echo "ERROR: Reconstruction failed!"
    exit 1
fi

echo "✅ Reconstruction completed!"
echo ""

# Step 2: Video Editing
echo "Step 2: Running Video Editing..."
echo "========================================="

PROMPT="oil painting in Van Gogh style"

python3 main_3dgsvid.py -s $SOURCE_DATA \
    --iteration=$ITERS \
    --use_dual \
    --group_size=$GNUM \
    --deform_type "multi" \
    --random_pts_num 60000 \
    --radius 1 3 \
    --tfloss_weight 2.0 \
    --editing_method="norecursive_single" \
    --initial_editor 2 \
    --prompt "$PROMPT" \
    --cate="sty" \
    --recursive_num 0 \
    --cuda_num=0

if [ $? -ne 0 ]; then
    echo "ERROR: Editing failed!"
    exit 1
fi

echo "✅ Video editing completed!"
echo ""

# Create output videos
echo "Creating output videos..."
OUTPUT_DIR="output/${VIDEO_NAME}"

if [ -d "$OUTPUT_DIR/test/ours_${ITERS}/renders" ]; then
    ffmpeg -framerate 10 -i "$OUTPUT_DIR/test/ours_${ITERS}/renders/%05d.png" \
           -c:v libx264 -pix_fmt yuv420p -crf 18 -y \
           "output/${VIDEO_NAME}_result.mp4" 2>&1 | tail -5
    
    echo "✓ Video created: output/${VIDEO_NAME}_result.mp4"
    ls -lh "output/${VIDEO_NAME}_result.mp4"
fi

echo ""
echo "========================================="
echo "🎉 SUCCESS! 🎉"
echo "Finished at: $(date)"
echo "========================================="
echo ""
echo "Results:"
find output -name "*.mp4" -exec ls -lh {} \;
echo ""

