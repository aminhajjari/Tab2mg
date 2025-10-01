#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=Table2Image_CVAE
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1        # Request one A100 GPU
#SBATCH --cpus-per-task=4             # 4 CPU cores
#SBATCH --mem=32G                     # 32 GB of memory
#SBATCH --time=06:00:00               # 6 hours for 50 epochs (increased from 4)
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=table2image_%j.out   # Output file with job ID
#SBATCH --error=table2image_%j.err    # Error file with job ID

# --- Environment Setup ---
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "=========================================="

# Define the path to your Python script
PYTHON_SCRIPT="table2image.py"

# !!! UPDATE THESE PATHS !!!
# Path to your CSV dataset
CSV_DATA_PATH="/project/def-arashmoh/shahab33/Msc/CSV/adult.data"

# Path where the model will be saved (include .pt extension)
SAVE_MODEL_PATH="/project/def-arashmoh/shahab33/Msc/Tab2mg/best_model.pt"

# Path to your image datasets (FashionMNIST and MNIST)
DATASET_ROOT="/project/def-arashmoh/shahab33/Msc/datasets"

# Navigate to the directory containing your script
SCRIPT_DIR="/project/def-arashmoh/shahab33/Msc"
cd $SCRIPT_DIR

echo "Working directory: $(pwd)"
echo "Script: $PYTHON_SCRIPT"
echo "CSV Data: $CSV_DATA_PATH"
echo "Model Save Path: $SAVE_MODEL_PATH"
echo "Dataset Root: $DATASET_ROOT"
echo "=========================================="

# Load necessary modules
module purge
module load python/3.10
module load cuda/11.8
module load scipy-stack  # For statsmodels and sklearn

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

# Verify required files exist
if [ ! -f "$CSV_DATA_PATH" ]; then
    echo "ERROR: CSV file not found at $CSV_DATA_PATH"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset root directory not found at $DATASET_ROOT"
    exit 1
fi

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
python $PYTHON_SCRIPT \
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
