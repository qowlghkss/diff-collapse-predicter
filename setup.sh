#!/bin/bash
# setup.sh - Professional environment and directory setup

echo "--- Initializing Diffusion Collapse Predictor (KISTI Silver Standard) ---"

# 1. Create directory structure
mkdir -p data/multiview model results figures src scripts

# 2. Check for Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10+."
    exit 1
fi

# 3. Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Initialize config if missing
if [ ! -f model/config.json ]; then
    echo "Creating default config.json..."
    cat <<EOF > model/config.json
{
  "early_t_end": 16,
  "collapse_percentile": 25,
  "logistic_regression": {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000
  },
  "random_seed": 42
}
EOF
fi

# 5. Summary
echo "Setup complete. Repository structure:"
ls -F data/ src/ model/ results/

echo "------------------------------------------------------------------------"
echo "To start, run: python src/data/split_data.py"
