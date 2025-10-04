#!/bin/bash

#=======================================================================
# Slurm Settings
#=======================================================================
# --- Resource Request ---
#SBATCH --account=def-arashmoh              # Your allocation account
#SBATCH --job-name=Table2Image_Robust       # A more descriptive job name
#SBATCH --nodes=1                           # Requesting one server node
#SBATCH --gpus-per-node=a100:1              # Requesting 1 A100 GPU
#SBATCH --cpus-per-task=4                   # Number of CPU cores for data loading
#SBATCH --mem=32G                           # Memory request (adjust if you get OOM errors)
#SBATCH --time=08:00:00                     # Increased time request for safety on first runs

# --- Job & Output Management ---
# Using Slurm's environment variables for more organized output files.
# This keeps logs inside a dedicated directory.
#SBATCH --output=/project/def-arashmoh/shahab33/Msc/job_logs/table2image_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/job_logs/table2image_%A.err
# Note: %A is the Slurm Job Array ID, which is better than %j for array jobs.
# For single jobs, it's similar to %j but is a good habit to use.

# --- Email Notifications ---
#SBATCH --mail-user=aminhjjr@gmail.com      # Your email
#SBATCH --mail-type=ALL                     # Receive emails for job start, end, and failure

#=======================================================================
# Environment Setup & Pre-run Checks (CRITICAL)
#=======================================================================
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "=========================================="

# --- Create Log Directory ---
# This command ensures the output/error directory exists before the job starts.
# Prevents an immediate job failure if the folder is missing.
LOG_DIR="/project/def-arashmoh/shahab33/Msc/job_logs"
mkdir -p "$LOG_DIR"
echo "Ensured log directory exists at: $LOG_DIR"

# --- Define Paths Flexibly ---
# Using the $PROJECT variable is good practice if it's set on your cluster.
# If not, the full path is fine.
# This makes the script easier to read and modify.
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
PYTHON_SCRIPT="$PROJECT_DIR/Tab2mg/run_vif.py"
CSV_DATA_PATH="$PROJECT_DIR/CSV/data.csv"
MODEL_SAVE_DIR="$PROJECT_DIR/OutOrgin"
DATASET_ROOT="$PROJECT_DIR/datasets"

# Create a unique model save path for this specific job
FINAL_MODEL_SAVE_PATH="$MODEL_SAVE_DIR/final_cvaemodel_${SLURM_JOB_ID}.pth"

# --- **CRITICAL** File Existence Checks ---
# This section will cause the job to fail early with a clear message
# if a required file or directory is missing.
echo "--- Verifying paths ---"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "FATAL ERROR: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi
if [ ! -f "$CSV_DATA_PATH" ]; then
    echo "FATAL ERROR: CSV file not found at $CSV_DATA_PATH"
    exit 1
fi
if [ ! -d "$DATASET_ROOT" ]; then
    echo "FATAL ERROR: Dataset root directory not found at $DATASET_ROOT"
    exit 1
fi
# Ensure the model save directory exists
mkdir -p "$MODEL_SAVE_DIR"
echo "All paths verified successfully."
echo "=========================================="

#=======================================================================
# Software Environment
#=======================================================================
echo "--- Loading modules ---"
module purge
module load StdEnv/2023
module load intel/2023.2.1
module load cuda/11.8
echo "Modules loaded:"
module list
echo "=========================================="

echo "--- Activating Python environment ---"
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "FATAL ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"
echo "Virtual environment activated."
echo "Python executable: $(which python)"
echo "=========================================="

# --- Sanity Checks for Python and PyTorch ---
echo "--- Checking Python packages ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available to PyTorch: {torch.cuda.is_available()}');"
if [ $? -ne 0 ]; then
    echo "FATAL ERROR: PyTorch check failed. Is it installed in your venv?"
    exit 1
fi
echo "=========================================="

#=======================================================================
# Execute the Python Script
#=======================================================================
echo "Starting Table2Image training..."
echo "Running command:"
echo "python \"$PYTHON_SCRIPT\" \\"
echo "  --csv \"$CSV_DATA_PATH\" \\"
echo "  --save_dir \"$FINAL_MODEL_SAVE_PATH\" \\"
echo "  --dataset_root \"$DATASET_ROOT\""
echo "=========================================="

# Execute the script
python "$PYTHON_SCRIPT" \
  --csv "$CSV_DATA_PATH" \
  --save_dir "$FINAL_MODEL_SAVE_PATH" \
  --dataset_root "$DATASET_ROOT"

# Capture the exit code of the Python script
exit_code=$?

echo "=========================================="
if [ $exit_code -eq 0 ]; then
  echo "Job completed successfully!"
  echo "Model saved to: $FINAL_MODEL_SAVE_PATH"
else
  echo "Job FAILED with exit code: $exit_code"
  echo "Check the error file for details: $LOG_DIR/table2image_${SLURM_JOB_ID}.err"
fi
echo "Job finished at: $(date)"
echo "=========================================="

exit $exit_code
