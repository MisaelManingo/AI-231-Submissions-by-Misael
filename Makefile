.PHONY: help install reproduce train export bench

help:
	@echo "AI 231 Machine Exercise 2 - Commands:"
	@echo "  make install    Install dependencies from requirements.txt"
	@echo "  make reproduce  Run full end-to-end download, training, export, and evaluation"
	@echo "  make train      Train BC-ResNet-1 and DS-CNN on A100 across 3 seeds (20 classes)"
	@echo "  make export     Export checkpoints to FP32 and INT8 ONNX"
	@echo "  make bench      Run local CPU benchmark"

install:
	pip install -r requirements.txt

reproduce:
	./reproduce.sh

train:
	python "ME2 - Voice Command Model/scripts/train_20class.py" --model both --seeds 42,1337,2026 --epochs 25

export:
	python "ME2 - Voice Command Model/scripts/export_20class_onnx.py"

bench:
	python "ME2 - Voice Command Model/scripts/bench_pi.py" --runs 200 --threads 4
