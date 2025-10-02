#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=Table2Image_CVAE
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/def-arashmoh/shahab33/Msc/OutOrgin/table2image_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/OutOrgin/table2image_%j.err

# --- Environment Setup ---
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "=========================================="

# Define the path to your Python script
PYTHON_SCRIPT="run_vif.py"

# !!! Your Final Paths !!!
CSV_DATA_PATH="/project/def-arashmoh/shahab33/Msc/CSV/data.csv"
SAVE_MODEL_PATH="/project/def-arashmoh/shahab33/Msc/OutOrgin/final_cvae_model_${SLURM_JOB_ID}.pth"
DATASET_ROOT="/project/def-arashmoh/shahab33/Msc/datasets"

# --- FIX 2: Corrected directory where run_vif.py is located ---
SCRIPT_DIR="/project/def-arashmoh/shahab33/Msc/Tab2mg"
cd "$SCRIPT_DIR"
# -------------------------------------------------------------------

echo "Working directory: $(pwd)"
echo "Script: $PYTHON_SCRIPT"
echo "CSV Data: $CSV_DATA_PATH"
echo "Model Save Path: $SAVE_MODEL_PATH"
echo "Dataset Root: $DATASET_ROOT"
echo "=========================================="

# --- FIX 1: Load necessary modules with correct dependencies ---
module purge
module load StdEnv/2023
module load intel/2023.2.1  # Required by cuda/11.8
module load cuda/11.8       # Now loads successfully
# Note: Removed incompatible module loads like scipy-stack.
# ---------------------------------------------------------------

echo "Loaded modules:"
module list
echo "=========================================="

# Activate your virtual environment
source /project/def-arashmoh/shahab33/Msc/venvMsc/bin/activate

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "=========================================="

# Verify required files exist (Keep these checks!)
if [ ! -f "$CSV_DATA_PATH" ]; then
    echo "ERROR: CSV file not found at $CSV_DATA_PATH"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset root directory not found at $DATASET_ROOT"
    exit 1
fi

# ... (Dataset warnings kept for completeness) ...
if [ ! -d "$DATASET_ROOT/FashionMNIST" ]; then
    echo "WARNING: FashionMNIST dataset not found at $DATASET_ROOT/FashionMNIST"
    echo "The script will attempt to download it."
fi

if [ ! -d "$DATASET_ROOT/MNIST" ]; then
    echo "WARNING: MNIST dataset not found at $DATASET_ROOT/MNIST"
    echo "The script will attempt to download it."
fi

echo "=========================================="
echo "Starting Table2Image training..."
echo "=========================================="

# Execute the Python script with its required arguments
python "$PYTHON_SCRIPT" \
    --csv "$CSV_DATA_PATH" \
    --save_dir "$SAVE_MODEL_PATH" \
    --dataset_root "$DATASET_ROOT"

exit_code=$?

echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "Job completed successfully!"
    echo "Model saved to: $SAVE_MODEL_PATH"
else
    echo "Job failed with exit code: $exit_code"
fi
echo "Job finished at: $(date)"
echo "=========================================="

exit $exit_code
