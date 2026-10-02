#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
EXPORTS_DIR="$BASE_DIR/exports/v3_94class"
mkdir -p "$EXPORTS_DIR"

PYTHON="/home/misael.andre.maningo/.conda/envs/AI_231_env/bin/python"

echo "================================================================================"
echo "🎯 STARTING CONSOLIDATED 94-CLASS VOICE COMMAND TRAINING & BENCHMARK PIPELINE"
echo "Device: NVIDIA A100-SXM4-40GB (GPU 7)"
echo "Pinned HF Revision: 6947f13073e57eb6ae67e7e2fc3680700b82aa13"
echo "================================================================================"

# 1. Train BC-ResNet-1 (3 Seeds: 42, 1337, 2026)
echo "[1/4] Training BC-ResNet-1 across 3 seeds (42, 1337, 2026)..."
$PYTHON "$SCRIPT_DIR/train_94class.py" --model bcresnet --epochs 25 --seeds 42 1337 2026

# 2. Train DS-CNN baseline (3 Seeds: 42, 1337, 2026)
echo "[2/4] Training DS-CNN baseline across 3 seeds (42, 1337, 2026)..."
$PYTHON "$SCRIPT_DIR/train_94class.py" --model dscnn --epochs 25 --seeds 42 1337 2026

# 3. Supplemental Synth Ablation (Seed 42)
echo "[3/4] Running Supplemental Synth Ablation (Seed 42)..."
$PYTHON "$SCRIPT_DIR/train_94class.py" --model bcresnet --epochs 25 --seeds 42 --use-supp-synth

# 4. Export ONNX Models & Profile with vcmbench/flops.py
echo "[4/4] Exporting FP32/INT8 ONNX and profiling FLOPs/MACs..."
$PYTHON "$SCRIPT_DIR/export_94class_onnx.py"

echo "================================================================================"
echo "✅ ALL TRAINING, EVALUATION, AND EXPORTS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
