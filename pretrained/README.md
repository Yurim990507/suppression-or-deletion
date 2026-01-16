---
license: mit
tags:
- machine-unlearning
- sparse-autoencoder
- vision-transformer
- interpretability
- restoration
---

# Suppression or Deletion: Pretrained Models

This repository contains pretrained models and SAE (Sparse Autoencoder) assets for testing SAE-based restoration on machine unlearned models.

**Main GitHub Repository**: [suppression-or-deletion](https://github.com/Yurim0507/suppression-or-deletion)

## Overview

These assets enable researchers to test whether unlearned models can be restored during inference using Sparse Autoencoder (SAE) features. The repository includes:

- **Original ViT-Base/16 models** trained on CIFAR-10 and Imagenette
- **SAE models** trained on layers 8, 9, 10 with TopK sparsity
- **Activation statistics** for normalization
- **Expert features** identified for each class

## Repository Contents

```
pretrained/
├── cifar10/
│   ├── vit_base_16_original.pth          # Original ViT-Base model (~350 MB)
│   ├── sae_layer8_k16.pt                 # SAE for layer 8, k=16 (~25 MB)
│   ├── sae_layer9_k16.pt                 # SAE for layer 9, k=16 (~25 MB)
│   ├── sae_layer10_k16.pt                # SAE for layer 10, k=16 (~25 MB)
│   ├── activations_layer8_stats.npy      # Normalization stats for layer 8 (<1 MB)
│   ├── activations_layer9_stats.npy      # Normalization stats for layer 9 (<1 MB)
│   ├── activations_layer10_stats.npy     # Normalization stats for layer 10 (<1 MB)
│   ├── expert_features_layer8_k16.pt     # Expert features for layer 8 (<1 MB)
│   ├── expert_features_layer9_k16.pt     # Expert features for layer 9 (<1 MB)
│   └── expert_features_layer10_k16.pt    # Expert features for layer 10 (<1 MB)
└── imagenette/
    ├── vit_base_16_original.pth          # Original ViT-Base model (~350 MB)
    ├── sae_layer8_k32.pt                 # SAE for layer 8, k=32 (~25 MB)
    ├── sae_layer9_k32.pt                 # SAE for layer 9, k=32 (~25 MB)
    ├── sae_layer10_k32.pt                # SAE for layer 10, k=32 (~25 MB)
    ├── activations_layer8_stats.npy      # Normalization stats for layer 8 (<1 MB)
    ├── activations_layer9_stats.npy      # Normalization stats for layer 9 (<1 MB)
    ├── activations_layer10_stats.npy     # Normalization stats for layer 10 (<1 MB)
    ├── expert_features_layer8_k32.pt     # Expert features for layer 8 (<1 MB)
    ├── expert_features_layer9_k32.pt     # Expert features for layer 9 (<1 MB)
    └── expert_features_layer10_k32.pt    # Expert features for layer 10 (<1 MB)
```

**Total size**: ~860 MB

## Dataset-Specific Configurations

| Dataset | Classes | TopK (k) | Expert Features per Class |
|---------|---------|----------|---------------------------|
| **CIFAR-10** | 10 | 16 | 20 (k×5/4) |
| **Imagenette** | 10 | 32 | 40 (k×5/4) |

## Quick Start

### Download All Files

```bash
# Using Hugging Face CLI (recommended)
pip install huggingface_hub
huggingface-cli download Yurim0507/suppression-or-deletion --local-dir ./pretrained --repo-type=model
```

```python
# Using Python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Yurim0507/suppression-or-deletion",
    local_dir="./pretrained",
    repo_type="model"
)
```

### Download Specific Dataset

```bash
# CIFAR-10 only
huggingface-cli download Yurim0507/suppression-or-deletion --include "cifar10/*" --local-dir ./pretrained --repo-type=model

# Imagenette only
huggingface-cli download Yurim0507/suppression-or-deletion --include "imagenette/*" --local-dir ./pretrained --repo-type=model
```

## Usage with Main Repository

1. **Clone the main repository**:
   ```bash
   git clone https://github.com/Yurim0507/suppression-or-deletion.git
   cd suppression-or-deletion
   ```

2. **Download pretrained assets** (using commands above)

3. **Prepare your unlearned model**:
   - Train an unlearned model using any unlearning method (CF-k, SALUN, SCRUB, etc.)
   - Save the checkpoint in `.pth` format

4. **Run restoration test**:
   ```bash
   # Test restoration on CIFAR-10 class 0 (airplane)
   python recovery_test.py \
       --dataset cifar10 \
       --unlearned_model path/to/your/unlearned_model.pth \
       --target_class 0 \
       --layer 9 \
       --alpha 1.0 2.0 5.0 10.0
   ```

5. **Simple demo script**:
   ```bash
   python demo.py \
       --dataset cifar10 \
       --unlearned_model path/to/your/unlearned_model.pth \
       --target_class 0
   ```

## File Formats

### Original Model (`vit_base_16_original.pth`)

PyTorch checkpoint containing ViT-Base/16 model trained on CIFAR-10 or Imagenette:
```python
{
    'model_state_dict': <OrderedDict>,  # Model weights
    'epoch': <int>,                      # Training epoch
    # ... other training metadata
}
```

### SAE Model (`sae_layer{8,9,10}_k{16,32}.pt`)

Sparse Autoencoder checkpoint:
```python
{
    'model_state_dict': <OrderedDict>,  # SAE weights
    'model_config': {
        'input_dim': 768,               # ViT hidden dimension
        'hidden_dim': 3072,             # SAE latent dimension (768×4)
        'k': 16,                        # TopK sparsity (16 for CIFAR-10, 32 for Imagenette)
        'activation': 'topk'            # Activation type
    },
    'pre_bias': <Tensor>,               # Pre-bias parameter
}
```

### Activation Statistics (`activations_layer{8,9,10}_stats.npy`)

Normalization statistics:
```python
{
    'patch_mean': <ndarray>,  # Mean of patch token activations
    'patch_std': <ndarray>,   # Std of patch token activations
}
```

### Expert Features (`expert_features_layer{8,9,10}_k{16,32}.pt`)

Class-specific expert features:
```python
{
    'class_experts_details': {
        0: [feature_id_1, feature_id_2, ...],  # Expert features for class 0
        1: [...],                               # Expert features for class 1
        ...
        9: [...]                                # Expert features for class 9
    }
}
```

**Expert feature selection criteria:**
- Top k×5/4 features per class (20 for CIFAR-10, 40 for Imagenette)
- Sorted by F1 score
- Common features (active in 7+ classes) excluded

## Training Details

### Original Models

- **Architecture**: ViT-Base/16 (google/vit-base-patch16-224)
- **Training**: Fine-tuned on CIFAR-10/Imagenette from pretrained ImageNet weights
- **Optimizer**: AdamW
- **Learning rate**: 1e-4
- **Epochs**: 20
- **Data augmentation**: RandomCrop, RandomHorizontalFlip

### SAE Models

- **Layers**: 8, 9, 10 (out of 12 ViT layers)
- **Architecture**: Overcomplete (768 → 3072 → 768)
- **Sparsity**: TopK activation
  - **CIFAR-10**: k=16 (only top 16 features active per sample)
  - **Imagenette**: k=32 (only top 32 features active per sample)
- **Training loss**: MSE reconstruction + L1 regularization
- **Training samples**: All training set activations (patch tokens only)

### Expert Features

- **Selection criteria**: Top k×5/4 features per class
  - **CIFAR-10**: 20 features per class (16×5/4)
  - **Imagenette**: 40 features per class (32×5/4)
- **Metrics**: F1 score, precision, recall
- **Filtering**: Common features (active in 7+ classes) excluded

## Restoration Method

Expert features are amplified by coefficient **α (alpha)** during inference:

- **α = 1.0**: No amplification (baseline)
- **α = 2.0**: Double the expert feature strength
- **α = 5.0**: 5x amplification
- **α = 10.0**: 10x amplification

The restoration uses **direct injection mode**, which requires the original model's activations.

## Citation

If you use these models in your research, please cite:

```bibtex
@article{suppression-or-deletion,
  title={Suppression or Deletion: Understanding Machine Unlearning via SAE-based Restoration},
  author={...},
  year={2024}
}
```

## License

MIT License - See main repository for details.

## Questions and Issues

For questions or issues, please open an issue on the [main GitHub repository](https://github.com/Yurim0507/suppression-or-deletion/issues).
