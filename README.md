# AI 231 Submissions by Misael Maningo

This repository contains Machine Exercises, projects, and assignments for **AI 231: Advanced Deep Learning** by **Misael Maningo**.

---

## 📂 Repository Structure

| Directory | Topic / Machine Exercise | Key Technologies | Status |
|---|---|---|---|
| [`ME1 - Einops/Einsum`](./ME1%20-%20Einops/Einsum/) | Custom 3-Layer CNN for MNIST Classification | PyTorch, `einops`, `torch.einsum`, First-Principles CNN | Completed (Test Acc: 98.99%) |
| [`ME2 - Voice Command Model`](./ME2%20-%20Voice%20Command%20Model/) | On-Device Tiny Voice Command & Personalized Wake Word Spotter on Raspberry Pi 5 | BC-ResNet-1, RobustWakeNet, ONNX Runtime INT8, Pure NumPy DSP, 20-Class Acoustic Intent & OOS Rejection | Completed (Test Acc: 83.62% ± 0.18%, FAR: 4.97%) |

---

## 🛠️ Machine Exercise Overview

### 1. [ME1: Einops & Einsum CNN](./ME1%20-%20Einops/Einsum/)
- Implements convolutional layers, spatial reductions, and dense projections strictly from first principles using tensor contractions and index manipulations.
- Verified on MNIST achieving **98.99% test accuracy**.

### 2. [ME2: Edge Voice Command Model & Wake Word Spotter](./ME2%20-%20Voice%20Command%20Model/)
- Complete, 100% on-device edge voice assistant deployed on **Raspberry Pi 5** using a **G-Mark Micro Go USB mic/speaker**.
- Zero PyTorch dependencies on device: pure NumPy Mel-Spectrogram extraction + ONNX Runtime CPU.
- **Personalized Wake Word Spotter ("Hey Raspberry")**: 27.4 KB INT8, 98.5% confidence on user takes, 0.00% on idle noise.
- **20-Class Voice Command Model (BC-ResNet-1)**: 110 KB INT8 (0.0685 M parameters), 83.62% ± 0.18% test accuracy across 121 unseen speakers on the multi-source Hugging Face benchmark, combined OOS FAR of 4.97% (at tuned $\tau^* = 0.77$).
- Full real-time smart-home state management with continuous ALSA audio capture, adaptive peak normalization, and VAD energy gating.
- Full multi-seed A100 training logs, DS-CNN baseline comparison, INT8 ONNX export, and one-command reproduction (`./reproduce.sh`).

---

## 👥 Contributors

- **Misael Maningo** ([@MisaelManingo](https://github.com/MisaelManingo)) - Student / Developer
- **Google Antigravity** - AI Assistant / Autonomous Coding Agent
