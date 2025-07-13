#!/bin/bash
# This script runs the benchmark for the given model and dataset.
# set -e

VENV_PATH = ".venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
        # Source the activate script to apply environment changes
        source "${VENV_PATH}/bin/activate"
        echo "uv virtual environment activated."
else
    echo "Error: uv virtual environment not found at ${VENV_PATH}."
    exit 1
fi

echo "Downloading Amazon Fine Food Reviews dataset..."
kaggle datasets download -d datasets/snap/amazon-fine-food-reviews
unzip amazon-fine-food-reviews.zip -d amazon-fine-food-reviews

echo "Running benchmark for model: $1 with embedding dimension: $2"
python benchmark.py \
    --path amazon-fine-food-reviews/Reviews.csv \
    --model $1 \
    --embedding_dim $2

echo "Benchmark completed."
