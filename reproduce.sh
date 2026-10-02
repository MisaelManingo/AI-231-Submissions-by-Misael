#!/usr/bin/env bash
set -e

echo "========================================================================"
echo "🔁 FULL REPRODUCIBILITY PIPELINE: AI231 ME2 VOICE COMMAND MODEL (32-CLASS)"
echo "Pinned HF Revision: 6947f13073e57eb6ae67e7e2fc3680700b82aa13"
echo "========================================================================"

PYTHON_BIN="${PYTHON:-python3}"
ME2_DIR="ME2 - Voice Command Model"

echo "[1/4] Preparing 32-class dataset from Hugging Face cache..."
$PYTHON_BIN "$ME2_DIR/scripts/prep_dataset_32class.py"

echo "[2/4] Training BC-ResNet-1 and baseline DS-CNN across 3 seeds on GPU..."
$PYTHON_BIN "$ME2_DIR/scripts/train_32class.py" --epochs 20 --loso_epochs 12

echo "[3/4] Exporting checkpoints to FP32 and INT8 ONNX..."
$PYTHON_BIN "$ME2_DIR/scripts/export_32class_onnx.py"
echo "[3/4] Packaging standalone deployment bundle (ME2-quickstart.zip)..."
$PYTHON_BIN "$ME2_DIR/scripts/make_quickstart_zip.py"

echo "[4/4] Verifying documentation math rendering..."
$PYTHON_BIN "$ME2_DIR/scripts/verify_readme_math.py" "$ME2_DIR/README.md"

echo "========================================================================"
echo "✅ REPRODUCIBILITY RUN COMPLETED SUCCESSFULLY!"
echo "Artifacts are stored in: $ME2_DIR/exports/v4_32class/"
echo "Quickstart bundle: $ME2_DIR/ME2-quickstart.zip"
echo "========================================================================"
