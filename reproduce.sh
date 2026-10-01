#!/usr/bin/env bash
set -e

echo "========================================================================"
echo "🔁 FULL REPRODUCIBILITY PIPELINE: AI231 ME2 VOICE COMMAND MODEL"
echo "========================================================================"

PYTHON_BIN="${PYTHON:-python3}"
ME2_DIR="ME2 - Voice Command Model"

echo "[1/4] Downloading and preparing 20-class dataset from Hugging Face..."
$PYTHON_BIN "$ME2_DIR/scripts/download_and_prep_dataset.py"

echo "[2/4] Training BC-ResNet-1 and baseline DS-CNN across 3 seeds on GPU..."
$PYTHON_BIN "$ME2_DIR/scripts/train_20class.py" --model both --seeds 42,1337,2026 --epochs 25

echo "[3/4] Exporting checkpoints to FP32 and INT8 ONNX..."
$PYTHON_BIN "$ME2_DIR/scripts/export_20class_onnx.py"

echo "[4/4] Running local benchmark validation..."
$PYTHON_BIN "$ME2_DIR/scripts/bench_pi.py" --runs 200 --threads 4

echo "========================================================================"
echo "✅ REPRODUCIBILITY RUN COMPLETED SUCCESSFULLY!"
echo "Artifacts are stored in: $ME2_DIR/exports/v2_20class/"
echo "========================================================================"
