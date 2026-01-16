import torch
import random
import numpy as np
from transformers import ViTConfig, ViTForImageClassification


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    model_name = model_name.lower()

    vit_map = {
        "vit_base_16":  "google/vit-base-patch16-224",
        "vit_large_16": "google/vit-large-patch16-224-in21k",
    }

    if model_name in vit_map:
        ckpt = vit_map[model_name]
        if pretrained:
            return ViTForImageClassification.from_pretrained(
                ckpt, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        cfg = ViTConfig.from_pretrained(ckpt)
        cfg.num_labels = num_classes
        return ViTForImageClassification(cfg)

    raise ValueError(
        f"Model '{model_name}' not available. "
        f"Choose from {list(vit_map.keys())}"
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
