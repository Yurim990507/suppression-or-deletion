# Pretrained Models

This directory should contain pretrained SAE models and original ViT models.

## Download from Hugging Face

Download all pretrained files from our Hugging Face repository:

```bash
pip install huggingface_hub
huggingface-cli download Yurim0507/suppression-or-deletion --local-dir ./pretrained --repo-type=model
```

## Contents

After downloading, this directory will contain:
- **CIFAR-10**: Original ViT model + 3 SAE models (layers 8, 9, 10) + stats + expert features
- **Imagenette**: Original ViT model + 3 SAE models (layers 8, 9, 10) + stats + expert features


## More Information

For detailed documentation about the pretrained models, file formats, and training details, visit:

**https://huggingface.co/Yurim0507/suppression-or-deletion**
