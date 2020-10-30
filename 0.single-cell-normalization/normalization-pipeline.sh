#!/bin/bash
#
# Normalize single cells using a variety of methods
#
# 1. Z-score normalize (standardize)
# 2. Spherize transform (ZCA-cor)
# 3a. Harmony transform (per plate)
# 3b. Harmony transform (all plates)
#
# We apply feature selection to all plates individually,
# and retain features that were selected in at least two plates.
#
# Note: For the Harmony transform, we also perform an experimental "inverse Harmony transform"

set -e

# Step 0 - Convert all notebooks to scripts
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/nbconverted *.ipynb

# Step 1 - Perform profiling pipeline for standardization and feature selection
jupyter nbconvert --to=html \
    --FilesWriter.build_directory=scripts/html \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=10000000 \
    --execute 0.profiling-pipeline.ipynb

# Step 2 - Apply spherize transform
jupyter nbconvert --to=html \
    --FilesWriter.build_directory=scripts/html \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=10000000 \
    --execute 1.spherize-batch-effect.ipynb
    
# Step 3 - Perform Harmony transforms
jupyter nbconvert --to=html \
    --FilesWriter.build_directory=scripts/html \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=10000000 \
    --execute 2.apply-harmony-batch-effect.ipynb
