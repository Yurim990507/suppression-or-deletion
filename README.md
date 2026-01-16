# Suppression or Deletion: A Restoration-Based Representation-Level Analysis of Machine Unlearning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


---

## Overview

As pretrained models are increasingly shared on the web, ensuring models can forget sensitive, copyrighted, or private information has become crucial. Current unlearning evaluations rely on **output-based metrics**, which cannot verify whether information is truly **deleted** or merely **suppressed** at the representation level.

This repository provides a **restoration-based analysis framework** using Sparse Autoencoders to:
- Identify class-specific expert features in intermediate layers
- Apply inference-time steering to restore unlearned information
- Quantitatively distinguish between suppression and deletion

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Yurim990507/suppression-or-deletion.git
cd suppression-or-deletion

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Download Pretrained Assets

Download the pretrained SAE models, expert features, and original model from Hugging Face:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download all pretrained files
huggingface-cli download Yurim0507/suppression-or-deletion --local-dir ./pretrained --repo-type=model
```

The files will be organized as:

```
pretrained/
├── cifar10/
│   ├── vit_base_16_original.pth          # Original ViT model
│   ├── sae_layer9_k16.pt                 # SAE model (layer 9, k=16)
│   ├── activations_layer9_stats.npy      # Normalization statistics
│   └── expert_features_layer9_k16.pt     # Expert features per class
└── imagenette/
    ├── vit_base_16_original.pth          # Original ViT model
    ├── sae_layer9_k32.pt                 # SAE model (layer 9, k=32)
    ├── activations_layer9_stats.npy      # Normalization statistics
    └── expert_features_layer9_k32.pt     # Expert features per class
```

**Note**: Dataset-specific SAE configurations:
- **CIFAR-10**: k=16 (TopK sparsity)
- **Imagenette**: k=32 (TopK sparsity)

### 2. Prepare Your Unlearned Model

Train an unlearned model using your preferred method (CF-k, SALUN, SCRUB, etc.) and save it as a `.pth` checkpoint.

**Example checkpoint format:**
```python
{
    'model_state_dict': model.state_dict(),
    # ... other optional keys
}
```

### 3. Run Restoration Test

#### Simple Demo

```bash
python demo.py \
    --dataset cifar10 \
    --unlearned_model path/to/your/unlearned_model.pth \
    --target_class 0
```

#### Full Control

```bash
python recovery_test.py \
    --dataset cifar10 \
    --unlearned_model path/to/your/unlearned_model.pth \
    --target_class 0 \
    --layer 9 \
    --alpha 1.0 5.0 10.0 \
    --save_dir ./results
```

### 4. View Results

Results are saved in the `--save_dir` directory:
- `restoration_class{X}.png`: Line plot showing restoration performance
- `restoration_class{X}_results.json`: Detailed numerical results

---


## Methodology

### Sparse Autoencoder (SAE)

We train a sparse autoencoder on ViT layer activations with the following architecture:

- **Input**: ViT hidden states (768-dim for ViT-Base)
- **Latent**: 768-dim (hidden_mul=1)
- **Activation**: TopK sparsity (only top K features active per sample)
- **Output**: Reconstructed hidden states

**Dataset-specific configurations:**
| Dataset | K value (TopK sparsity) |
|---------|-------------------------|
| CIFAR-10 | 16 |
| Imagenette | 32 |

**Restoration mode:**
- `direct_injection`: Gradual addition method (default, used in experiments)
  - Only target class samples are restored
  - Non-target samples remain untouched

---

## Datasets

This repository supports **CIFAR-10** and **Imagenette** datasets used in our experiments.



