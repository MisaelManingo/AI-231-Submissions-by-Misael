# AI 231 Submissions by Misael Maningo

This repository contains Machine Exercises, projects, and assignments for **AI 231: Advanced Deep Learning** by **Misael Maningo**.

---

## 📂 Repository Structure

| Directory | Topic / Machine Exercise | Key Technologies | Status |
|---|---|---|---|
| [`ME1 - Einops/Einsum`](./ME1%20-%20Einops/Einsum/) | Custom 3-Layer CNN for MNIST Classification | PyTorch, `einops`, `torch.einsum`, First-Principles CNN | Completed (Test Acc: 98.99%) |

---

## 🛠️ Environment Setup

To reproduce the experiments in this repository, set up the `AI_231_env` conda environment:

```bash
# Create and activate environment
conda env create -f "ME1 - Einops/Einsum/environment.yml"
conda activate AI_231_env

# Register kernel for Jupyter
python -m ipykernel install --user --name AI_231_env --display-name "Python (AI_231_env)"
```

