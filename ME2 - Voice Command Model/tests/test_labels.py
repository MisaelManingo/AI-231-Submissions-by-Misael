#!/usr/bin/env python3
"""
Regression test for label mapping, ONNX model output dimensions,
and idx2label round-tripping with labels_94.json.
"""

import os
import sys
import json
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from demo_rpi5 import RPi5VoiceAssistant, SLOT_LABEL_MAP, SLOT_TYPE_MAP


def test_labels_and_model():
    print("=" * 65)
    print("🧪 RUNNING LABEL REGRESSION TEST (tests/test_labels.py)")
    print("=" * 65)

    labels_path = os.path.join(BASE_DIR, "exports/labels_94.json")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(BASE_DIR, "data/labels_94.json")
    assert os.path.exists(labels_path), f"labels_94.json not found at {labels_path}"

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    # 1. Initialize assistant with no meter
    assistant = RPi5VoiceAssistant(no_meter=True)

    # 2. Check label count
    idx2label = assistant.idx2label
    assert len(idx2label) == 94, f"Expected 94 labels, got {len(idx2label)}"
    print(f"✅ Label count is {len(idx2label)} (0..93)")

    # 3. Check every index 0..93 resolves without KeyError
    for i in range(94):
        assert i in idx2label, f"Missing index {i} in idx2label"
        label = idx2label[i]
        assert isinstance(label, str) and len(label) > 0, f"Invalid label for index {i}: {label}"

    assert idx2label[93] == "OUT_OF_SCOPE", f"Expected index 93 to be OUT_OF_SCOPE, got {idx2label[93]}"
    print("✅ All indices 0..93 resolve without KeyError, index 93 is OUT_OF_SCOPE")

    # 4. Check ONNX model output size equals label count
    vcm_path = os.path.join(BASE_DIR, "exports/bcresnet_94class_int8.onnx")
    assert os.path.exists(vcm_path), f"ONNX model not found at {vcm_path}"
    sess = ort.InferenceSession(vcm_path, providers=["CPUExecutionProvider"])
    vcm_output_dim = sess.get_outputs()[0].shape[-1]
    assert vcm_output_dim == len(idx2label), (
        f"ONNX model output dimension ({vcm_output_dim}) != idx2label count ({len(idx2label)})"
    )
    print(f"✅ Model output size ({vcm_output_dim}) matches label count ({len(idx2label)})")

    # 5. Check round-trip with labels_94.json classes
    classes = labels_data["classes"]
    assert len(classes) == 94, f"Expected 94 classes in labels_94.json, got {len(classes)}"

    for c in classes:
        idx = int(c["index"])
        intent = c.get("intent", "")
        slot = c.get("slot", "")
        label = idx2label[idx]

        if intent == "OUT_OF_SCOPE" or idx == 93:
            assert label == "OUT_OF_SCOPE", f"Index {idx} expected OUT_OF_SCOPE, got {label}"
        else:
            meta = assistant.slot_meta[label]
            assert meta["intent"] == intent, (
                f"Index {idx}: meta intent {meta['intent']} != expected {intent}"
            )
            if slot:
                assert meta["slot_value"] == slot, (
                    f"Index {idx}: meta slot_value {meta['slot_value']} != expected {slot}"
                )
            else:
                assert meta["slot_value"] is None, (
                    f"Index {idx}: meta slot_value should be None, got {meta['slot_value']}"
                )

    print("✅ idx2label round-trips with labels_94.json classes, intents, and slots")

    # 6. Check safe fallback with .get() for unknown / out-of-range indices
    for bad_idx in [-1, 94, 999]:
        fallback_label = idx2label.get(bad_idx, "OUT_OF_SCOPE")
        assert fallback_label == "OUT_OF_SCOPE", (
            f"Expected fallback 'OUT_OF_SCOPE' for index {bad_idx}, got {fallback_label}"
        )

    print("✅ .get() fallback handles out-of-range indices safely without crashing")
    print("=" * 65)
    print("🎉 ALL REGRESSION TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_labels_and_model()
