#!/bin/bash

#=======================================================================
# SLURM BATCH SCRIPT - Table2Image on ALL Datasets
#=======================================================================
# This script processes all datasets in the tabularDataset directory
# Author: Amin
# Date: 2025
#=======================================================================

#=======================================================================
# Slurm Settings
#=======================================================================
# --- Resource Request ---
#SBATCH --account=def-arashmoh
#SBATCH --job-name=Table2Image_ALL_Datasets
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00                     # 48 hours for ALL datasets

# --- Job & Output Management ---
#SBATCH --output=/project/def-arashmoh/shahab33/Msc/job_logs/batch_all_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/job_logs/batch_all_%A.err

# --- Email Notifications ---
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Environment Setup
#=======================================================================
echo "=========================================="
echo "BATCH PROCESSING - ALL DATASETS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "=========================================="

# --- Define Paths ---
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
OUTPUT_DIR="$PROJECT_DIR/batch_results_${SLURM_JOB_ID}"
BATCH_SCRIPT="$PROJECT_DIR/Tab2mg/run_all_datasets.py"
MAIN_SCRIPT="$PROJECT_DIR/Tab2mg/Am_v2.py"  # Your updated Am_v2.py
DATASET_ROOT="$PROJECT_DIR/datasets"  # For MNIST/FashionMNIST

# --- Create necessary directories ---
LOG_DIR="/project/def-arashmoh/shahab33/Msc/job_logs"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATASET_ROOT"

echo "Paths configured:"
echo "  Datasets: $DATASETS_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Batch script: $BATCH_SCRIPT"
echo "  Main script: $MAIN_SCRIPT"
echo "=========================================="

# --- Verify Critical Files ---
echo "Verifying files..."
if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ FATAL: Datasets directory not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ FATAL: Batch script not found: $BATCH_SCRIPT"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ FATAL: Main script not found: $MAIN_SCRIPT"
    exit 1
fi

# Count datasets
DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Found $DATASET_COUNT dataset folders"
echo "=========================================="

#=======================================================================
# Software Environment
#=======================================================================
echo "Loading modules..."
module purge
module load StdEnv/2023
module load intel/2023.2.1
module load cuda/11.8

echo "Activating virtual environment..."
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "❌ FATAL: Virtual environment not found: $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

echo "Python: $(which python)"
echo "PyTorch check:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"
echo "=========================================="

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo "Starting batch processing of ALL datasets..."
echo "This will process $DATASET_COUNT datasets"
echo "Estimated time: 1-2 hours per dataset"
echo "=========================================="

# Run the batch processor
python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --script_path "$MAIN_SCRIPT" \
    --epochs 50 \
    --batch_size 64 \
    --dataset_root "$DATASET_ROOT" \
    --timeout 7200

exit_code=$?

echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ BATCH PROCESSING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "  - results_all_datasets.csv (main results)"
    echo "  - results_latex.txt (for paper)"
    echo "  - Individual model files in subdirectories"
    echo ""
    echo "To view results:"
    echo "  cat $OUTPUT_DIR/results_all_datasets.csv"
else
    echo "❌ BATCH PROCESSING FAILED (exit code: $exit_code)"
    echo "Check error log: $LOG_DIR/batch_all_${SLURM_JOB_ID}.err"
fi
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

exit $exit_code
