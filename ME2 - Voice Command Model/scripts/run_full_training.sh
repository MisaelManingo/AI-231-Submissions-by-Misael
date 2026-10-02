#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
EXPORTS_DIR="$BASE_DIR/exports/v4_32class"
mkdir -p "$EXPORTS_DIR"

PYTHON="${PYTHON:-python3}"

echo "================================================================================"
echo "🎯 STARTING CONSOLIDATED 32-CLASS VOICE COMMAND TRAINING & BENCHMARK PIPELINE"
echo "Pinned HF Revision: 6947f13073e57eb6ae67e7e2fc3680700b82aa13"
echo "================================================================================"

# 1. Dataset Preparation & Verification
echo "[1/4] Preparing dataset and verifying disjointness assertions..."
$PYTHON "$SCRIPT_DIR/prep_dataset_32class.py"

# 2. Train BC-ResNet-1 and DS-CNN across 3 seeds (42, 1337, 2026) with LOSO CV
echo "[2/4] Running LOSO CV, rejection tuning, and 3-seed training..."
$PYTHON "$SCRIPT_DIR/train_32class.py" --epochs 20 --loso_epochs 12

# 3. Export ONNX Models & Profile with vcmbench/flops.py
echo "[3/4] Exporting FP32/INT8 ONNX and profiling FLOPs/MACs..."
$PYTHON "$SCRIPT_DIR/export_32class_onnx.py"

# 4. Package Quickstart Zip
echo "[4/4] Building standalone deployment bundle (ME2-quickstart.zip)..."
$PYTHON "$SCRIPT_DIR/make_quickstart_zip.py"

echo "================================================================================"
echo "✅ ALL 32-CLASS TRAINING, EVALUATION, AND EXPORTS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
