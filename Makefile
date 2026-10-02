.PHONY: help install reproduce train export quickstart verify

help:
	@echo "AI 231 Machine Exercise 2 - Commands:"
	@echo "  make install    Install dependencies from requirements.txt"
	@echo "  make reproduce  Run full end-to-end download, training, export, and evaluation"
	@echo "  make train      Train BC-ResNet-1 and DS-CNN on A100 across 3 seeds (32 classes)"
	@echo "  make export     Export checkpoints to FP32 and INT8 ONNX"
	@echo "  make quickstart Build standalone ME2-quickstart.zip bundle"
	@echo "  make verify     Verify README math formatting"

install:
	pip install -r requirements.txt

reproduce:
	./reproduce.sh

train:
	bash "ME2 - Voice Command Model/scripts/run_full_training.sh"

export:
	python "ME2 - Voice Command Model/scripts/export_32class_onnx.py"

quickstart:
	python "ME2 - Voice Command Model/scripts/make_quickstart_zip.py"

actions-quickstart:
	python "ME2 - Voice Command Model/scripts/make_actions_zip.py"

verify:
	python "ME2 - Voice Command Model/scripts/verify_readme_math.py" "ME2 - Voice Command Model/README.md"
