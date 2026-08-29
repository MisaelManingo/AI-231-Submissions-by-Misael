# Machine Exercise 1: Custom 3-Layer CNN for MNIST Classification using Einops & Einsum

**Course:** AI 231 - Advanced Deep Learning  
**Authors / Contributors:**  
- **Misael Maningo** (Student / Developer)  
- **Google Antigravity** (AI Assistant / Pair Programmer)  
**Environment:** `AI_231_env`  
**Notebook:** [`ME1_Einops_Einsum_MNIST.ipynb`](./ME1_Einops_Einsum_MNIST.ipynb)

---

## 📌 Executive Summary
In this Machine Exercise, we implement all layers and operations of a **3-Layer Convolutional Neural Network (CNN)** completely from first principles using tensor operations, `einops`, and `torch.einsum`.

**No pre-made CNN modules** (`nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.Flatten`) from packages are utilized. The entire neural pipeline—from receptive field unfolding to multi-dimensional tensor contraction, spatial dimension rearrangement, and affine transformations—is custom-built and fully differentiable.

---

## 🔬 Mathematical & Tensor Formulations

### 1. Custom 2D Convolution (`EinopsConv2d`)
- **Input Tensor**: $X \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$
- **Weight Tensor**: $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$
- **Bias Tensor**: $b \in \mathbb{R}^{C_{\text{out}}}$
- **Local Receptive Field Patches**: $P = \text{unfold}(X) \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{out}} \times W_{\text{out}} \times K_h \times K_w}$
- **Einstein Summation Contraction**:
  $$\text{Out}_{b, o, h, w} = \sum_{c=0}^{C_{\text{in}}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} P_{b, c, h, w, i, j} \cdot W_{o, c, i, j} + b_o$$
  ```python
  out = torch.einsum('b c h w i j, o c i j -> b o h w', patches, self.weight)
  out = out + rearrange(self.bias, 'o -> 1 o 1 1')
  ```

### 2. Custom 2D Max Pooling (`EinopsMaxPool2d`)
- **Spatial Window Rearrangement**:
  $$\text{patches} = \text{rearrange}(P, \text{'b c h w p1 p2 -> b c h w (p1 p2)'})$$
- **Window Reduction**:
  $$\text{Out}_{b, c, h, w} = \max_{k \in \{0, \dots, p_1 p_2 - 1\}} \text{patches}_{b, c, h, w, k}$$

### 3. Custom Linear Layer (`EinopsLinear`)
- **Matrix Multiplication as Tensor Contraction**:
  $$\text{Out}_{b, o} = \sum_{i=0}^{I-1} X_{b, i} \cdot W_{o, i} + b_o$$
  ```python
  out = torch.einsum('bi, oi -> bo', x, self.weight) + rearrange(self.bias, 'o -> 1 o')
  ```

### 4. Custom Flatten (`EinopsFlatten`)
- **Spatial / Channel Flattening**:
  ```python
  rearrange(x, 'b c h w -> b (c h w)')
  ```

### 5. Custom Activation (`EinopsReLU`)
- **Element-wise Clamping**:
  $$\text{ReLU}(x) = \max(0, x) = \text{clamp}(x, \min=0)$$

---

## 🏗️ 3-Layer CNN Architecture

| Layer | Type | Input Shape | Configuration / Kernel | Output Shape | Parameters |
|---|---|---|---|---|---|
| **Layer 1** | `EinopsConv2d` + `EinopsReLU` + `EinopsMaxPool2d` | $B \times 1 \times 28 \times 28$ | $1 \to 16$, $k=3$, $p=1$, pool $2\times 2$ | $B \times 16 \times 14 \times 14$ | 160 |
| **Layer 2** | `EinopsConv2d` + `EinopsReLU` + `EinopsMaxPool2d` | $B \times 16 \times 14 \times 14$ | $16 \to 32$, $k=3$, $p=1$, pool $2\times 2$ | $B \times 32 \times 7 \times 7$ | 4,640 |
| **Layer 3** | `EinopsConv2d` + `EinopsReLU` + `EinopsMaxPool2d` | $B \times 32 \times 7 \times 7$ | $32 \to 64$, $k=3$, $p=1$, pool $2\times 2$ | $B \times 64 \times 3 \times 3$ | 18,496 |
| **Head** | `EinopsFlatten` + `EinopsLinear` | $B \times 64 \times 3 \times 3$ | Flatten ($576$) $\to$ Linear ($576 \to 10$) | $B \times 10$ | 5,770 |
| **Total** | | | | | **29,066** |

---

## 📊 Experimental Results (5 Epochs)

The model was trained for 5 epochs using Adam optimizer ($\text{lr} = 0.001$) and Cross-Entropy Loss on MNIST ($60,000$ training images, $10,000$ test images):

| Epoch | Training Loss | Training Accuracy (%) | Test Loss | Test Accuracy (%) | Epoch Time |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.1735 | 94.80% | 0.0485 | 98.32% | 25.7s |
| **2** | 0.0483 | 98.47% | 0.0310 | 98.98% | 25.3s |
| **3** | 0.0344 | 98.88% | 0.0330 | 98.88% | 23.6s |
| **4** | 0.0264 | 99.11% | 0.0257 | 99.18% | 23.4s |
| **5** | 0.0214 | 99.29% | 0.0348 | **98.99%** | 24.9s |

- **Final Test Split Accuracy**: **98.99%** (9,899 / 10,000 correctly classified)
- **Peak Test Accuracy**: **99.18%** (Epoch 4)
- **Total Training Duration**: ~122 seconds

---

## 🖼️ 4x4 Prediction Sample Grid
16 random test images were sampled and evaluated. Ground truth (GT) and predicted labels (Pred) are rendered alongside visual verification in the notebook.

---

## 🚀 How to Run

### Using the Conda Environment:
```bash
conda activate AI_231_env
jupyter notebook "ME1 - Einops/Einsum/ME1_Einops_Einsum_MNIST.ipynb"
```

