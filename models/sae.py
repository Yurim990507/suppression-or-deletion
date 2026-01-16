import torch
import torch.nn as nn
import torch.nn.functional as F


class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        threshold = topk_values[..., -1:]  
        mask = (x >= threshold).float()
        result = F.relu(x) * mask
        return result


class Autoencoder(nn.Module):
    def __init__(self, n_inputs: int, n_latents: int, activation: nn.Module, pre_bias_init: torch.Tensor):
        super().__init__()

        self.pre_bias = nn.Parameter(pre_bias_init.clone())
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.init_weights()
        self.register_buffer("neuron_activity", torch.zeros(n_latents, dtype=torch.long))

    @torch.no_grad()
    def init_weights(self):
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.decoder.weight.data /= self.decoder.weight.data.norm(dim=0, keepdim=True)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def forward(self, x: torch.Tensor):
        x_centered = x - self.pre_bias
        latents_pre_act = self.encoder(x_centered) + self.latent_bias
        latents = self.activation(latents_pre_act)

        reconstructed = self.decoder(latents) + self.pre_bias
    
        if self.training:
            with torch.no_grad():
                active_in_batch = (latents > 0).any(dim=0)
                self.neuron_activity[active_in_batch] += 1

        return reconstructed, latents

    def reset_neuron_activity(self):
        self.neuron_activity.zero_()
