#!/bin/bash

#=======================================================================
# SLURM BATCH SCRIPT - Process ALL Datasets in tabularDataset/
#=======================================================================
# This script processes all 67+ datasets automatically
# Author: Shahab (aminhajjr@gmail.com)
# Updated: 2024
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=Table2Image_ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/job_logs/batch_all_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/job_logs/batch_all_%A.err

#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration - UPDATE THESE PATHS IF NEEDED
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
TAB2MG_DIR="$PROJECT_DIR/Tab2mg"

# Input/Output paths
DATASETS_DIR="$PROJECT_DIR/tabularDataset"          # Your 67+ dataset folders
OUTPUT_DIR="$PROJECT_DIR/ALL_RESULTS_${SLURM_JOB_ID}"  # Where results go
MNIST_ROOT="$PROJECT_DIR/datasets"                   # For MNIST/FashionMNIST

# Script paths (based on your GitHub structure)
BATCH_SCRIPT="$TAB2MG_DIR/run_all_datasets.py"      # Batch processor
MAIN_SCRIPT="$TAB2MG_DIR/run_vif.py"                # Your main training script

# Virtual environment
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"

# Training parameters
EPOCHS=50
BATCH_SIZE=64
TIMEOUT=7200  # 2 hours per dataset

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "BATCH PROCESSING ALL DATASETS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

#=======================================================================
# Create Directories
#=======================================================================
echo "Setting up directories..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MNIST_ROOT"
mkdir -p "/project/def-arashmoh/shahab33/Msc/job_logs"

echo "Directories created:"
echo "  Output: $OUTPUT_DIR"
echo "  MNIST: $MNIST_ROOT"

#=======================================================================
# Verify Files Exist
#=======================================================================
echo "=========================================="
echo "Verifying paths..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets directory not found!"
    echo "   Expected: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ ERROR: Batch script not found!"
    echo "   Expected: $BATCH_SCRIPT"
    echo ""
    echo "   Solution: Upload run_all_datasets.py to $TAB2MG_DIR/"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main script not found!"
    echo "   Expected: $MAIN_SCRIPT"
    echo ""
    echo "   Did you mean a different script?"
    echo "   Files in Tab2mg/:"
    ls -la "$TAB2MG_DIR"/*.py
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found!"
    echo "   Expected: $VENV_PATH"
    exit 1
fi

# Count datasets
DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo ""
echo "✅ All paths verified!"
echo "   Found $DATASET_COUNT dataset folders"
echo "=========================================="

#=======================================================================
# Load Software Environment
#=======================================================================
echo "Loading modules..."
module purge
module load StdEnv/2023
module load intel/2023.2.1
module load cuda/11.8

echo ""
echo "Activating Python environment..."
source "$VENV_PATH"

echo "  Python: $(which python)"
echo ""
echo "Checking PyTorch..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: PyTorch check failed!"
    exit 1
fi

echo "=========================================="

#=======================================================================
# Show Dataset Preview
#=======================================================================
echo "Dataset folders to process:"
find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | head -10 | while read dir; do
    folder_name=$(basename "$dir")
    file_count=$(find "$dir" -type f \( -name "*.csv" -o -name "*.arff" -o -name "*.data" \) | wc -l)
    echo "  - $folder_name ($file_count data file(s))"
done

if [ $DATASET_COUNT -gt 10 ]; then
    echo "  ... and $((DATASET_COUNT - 10)) more datasets"
fi
echo "=========================================="

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo ""
echo "🚀 STARTING BATCH PROCESSING"
echo "=========================================="
echo "This will process $DATASET_COUNT datasets"
echo "Estimated time: ~1 hour per dataset"
echo "Total estimated time: ~$((DATASET_COUNT / 24)) days"
echo ""
echo "Configuration:"
echo "  Epochs per dataset: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Timeout per dataset: $((TIMEOUT / 3600)) hours"
echo "=========================================="
echo ""

# Run the batch processor
python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --script_path "$MAIN_SCRIPT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_root "$MNIST_ROOT" \
    --timeout $TIMEOUT

exit_code=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "BATCH PROCESSING COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $exit_code"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✅ SUCCESS!"
    echo ""
    echo "Results location:"
    echo "  $OUTPUT_DIR/"
    echo ""
    echo "Generated files:"
    echo "  📊 results_all_datasets.csv  - Main results table"
    echo "  📄 results_latex.txt         - LaTeX table for paper"
    echo "  📁 [dataset_name]/           - Individual models"
    echo ""
    echo "To view results:"
    echo "  cat $OUTPUT_DIR/results_all_datasets.csv"
    echo ""
    echo "To copy results to your local machine:"
    echo "  scp -r shahab33@narval.alliancecan.ca:$OUTPUT_DIR/ ."
else
    echo "❌ FAILED (exit code: $exit_code)"
    echo ""
    echo "Check error log:"
    echo "  cat /project/def-arashmoh/shahab33/Msc/job_logs/batch_all_${SLURM_JOB_ID}.err"
    echo ""
    echo "Partial results may still be in:"
    echo "  $OUTPUT_DIR/"
fi

echo "=========================================="
exit $exit_code
