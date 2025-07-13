#!/bin/bash
# This script runs the benchmark for the given model and dataset.
# set -e

VENV_PATH="../.venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
        # Source the activate script to apply environment changes
        source "${VENV_PATH}/bin/activate"
        echo "uv virtual environment activated."
else
    echo "Error: uv virtual environment not found at ${VENV_PATH}."
    exit 1
fi

echo "Inflating Amazon Reviews dataset..."
echo "Make sure you have downloaded the dataset from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews and placed it in the current directory as 'amazon-reviews.zip'."
unzip amazon-reviews.zip -d amazon-reviews

echo "Running benchmark for model: $1 with embedding dimension: $2"
python benchmark.py \
    --path amazon-reviews/Reviews.csv \
    --model $1 \
    --embedding_dim $2

echo "Cleaning up..."
rm -rf amazon-reviews
echo "Benchmark completed."
