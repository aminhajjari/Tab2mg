#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=Table2Image_CVAE  # A descriptive name for your job
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1             # Request 1 GPU
#SBATCH --cpus-per-task=4             # 4 CPU cores (A good balance for one GPU)
#SBATCH --mem=32G                     # 32 GB of memory
#SBATCH --time=04:00:00               # Request 4 hours (adjust as needed for 50 epochs)
#SBATCH --mail-user=<aminhjjr@gmail.com> # Replace with your email address
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=a100:1        # Specifically request one A100 GPU

# --- Environment Setup ---
# The code assumes your Python script is named 'table2image.py'
PYTHON_SCRIPT="table2image.py"

# Define the path to your CSV dataset and the desired saving directory
# !!! REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL PATHS !!!
CSV_DATA_PATH="/project/def-arashmoh/shahab33/data/your_dataset.csv" 
SAVE_MODEL_PATH="/project/def-arashmoh/shahab33/Rohollah/projects/FeDK2P/models/table2image_best_model"

# Navigate to the directory containing your project/script
# Replace with the actual path to your script's directory
SCRIPT_DIR="/project/def-arashmoh/shahab33/Rohollah/projects/FeDK2P/FeDK2P" 
cd $SCRIPT_DIR

# Load necessary modules
module purge
module load python/3.10  # Use a specific, recent Python version
module load cuda/11.8    # Load a compatible CUDA version for A100

# Activate your Conda/Venv environment
source /project/def-arashmoh/shahab33/Rohollah/projects/FeDK2P/FeDK2P/fedk2p/bin/activate

# --- Execution ---
echo "Starting Table2Image training with A100 GPU..."
echo "Running script: $PYTHON_SCRIPT"
echo "Using CSV: $CSV_DATA_PATH"
echo "Saving model to: $SAVE_MODEL_PATH"

# Execute the Python script with its required arguments
python $PYTHON_SCRIPT \
    --csv "$CSV_DATA_PATH" \
    --save_dir "$SAVE_MODEL_PATH"

echo "Job finished."
