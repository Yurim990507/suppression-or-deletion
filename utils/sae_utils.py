import os
import re
import torch
import numpy as np
from models.sae import Autoencoder, TopK

def load_activation_stats(stats_path):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Activation stats not found: {stats_path}")

    if stats_path.endswith('.pt'):
        stats_dict = torch.load(stats_path, map_location='cpu', weights_only=False)
    else:
        stats_dict = np.load(stats_path, allow_pickle=True).item()

    return stats_dict


def load_sae_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAE model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        elif 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("SAE config not found in checkpoint")
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        input_dim = checkpoint['encoder.weight'].shape[1]
        hidden_dim = checkpoint['encoder.weight'].shape[0]

        k_match = re.search(r'_k(\d+)', os.path.basename(model_path))
        k = int(k_match.group(1)) if k_match else 32

        config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'k': k,
            'activation': 'batch_topk'
        }

    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    k = config['k']
    activation_type = config.get('activation', 'topk')

    if activation_type.lower() in ['topk', 'top_k']:
        activation_fn = TopK(k=k)
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}. Only 'topk' is supported.")

    pre_bias_init = checkpoint.get('pre_bias', torch.zeros(input_dim))

    sae_model = Autoencoder(input_dim, hidden_dim, activation_fn, pre_bias_init)
    sae_model.load_state_dict(state_dict)
    sae_model.to(device)
    sae_model.eval()

    return sae_model, config
