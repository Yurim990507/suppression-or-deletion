import torch

class BaseSAEHook:
    def __init__(self, sae_model, target_layer, activation_stats):
        self.sae_model = sae_model
        self.target_layer = target_layer
        self.activation_stats = activation_stats

        self._load_normalization_stats()

    def _load_normalization_stats(self):
        device = self.sae_model.pre_bias.device
        stats = self.activation_stats

        if 'patch_mean' in stats and 'patch_std' in stats:
            self.patch_mean = self._to_tensor(stats['patch_mean'], device)
            self.patch_std = self._to_tensor(stats['patch_std'], device)

            if 'cls_mean' in stats and 'cls_std' in stats:
                self.cls_mean = self._to_tensor(stats['cls_mean'], device)
                self.cls_std = self._to_tensor(stats['cls_std'], device)
                self.use_token_aware = True
            else:
                self.use_token_aware = False

        elif 'cls' in stats and 'patch' in stats:
            self.cls_mean = self._to_tensor(stats['cls']['mean'], device)
            self.cls_std = self._to_tensor(stats['cls']['std'], device)
            self.patch_mean = self._to_tensor(stats['patch']['mean'], device)
            self.patch_std = self._to_tensor(stats['patch']['std'], device)
            self.use_token_aware = True

        elif 'mean' in stats and 'std' in stats:
            global_mean = self._to_tensor(stats['mean'], device)
            global_std = self._to_tensor(stats['std'], device)
            self.cls_mean = self.patch_mean = global_mean
            self.cls_std = self.patch_std = global_std
            self.use_token_aware = False
        else:
            raise KeyError(f"Unknown activation_stats structure. Keys: {list(stats.keys())}")

    def _to_tensor(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return torch.from_numpy(data).to(device)

    def split_tokens(self, activations):
        cls_tokens = activations[:, 0:1, :]
        patch_tokens = activations[:, 1:, :]
        return cls_tokens, patch_tokens

    def normalize_patches(self, patch_tokens):
        batch_size, num_patches, hidden_dim = patch_tokens.shape
        patch_flat = patch_tokens.reshape(-1, hidden_dim)
        normalized = (patch_flat - self.patch_mean) / self.patch_std
        return normalized

    def denormalize_patches(self, normalized_patches):
        return normalized_patches * self.patch_std + self.patch_mean

    def reconstruct_output(self, cls_tokens, patch_tokens_reconstructed, original_shape):
        batch_size, seq_len, hidden_dim = original_shape
        patch_reconstructed = patch_tokens_reconstructed.view(batch_size, seq_len - 1, hidden_dim)
        return torch.cat([cls_tokens, patch_reconstructed], dim=1)

    def __call__(self, module, input, output):
        raise NotImplementedError("Subclass must implement __call__ method")


class SAERestorationHook(BaseSAEHook):
    def __init__(self, sae_model, target_layer, activation_stats, expert_features,
                 original_model, device, alpha=1.0, restoration_mode='add', target_class=None):
        super().__init__(sae_model, target_layer, activation_stats)
        self.expert_features = set(expert_features)
        self.alpha = alpha
        self.device = device
        self.restoration_mode = restoration_mode
        self.original_model = original_model
        self.current_input = None
        self.target_class = target_class
        self.current_labels = None

    def set_input(self, input_images, labels=None):
        self.current_input = input_images
        self.current_labels = labels

    def __call__(self, module, input_data, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        batch_size, seq_len, hidden_dim = hidden_states.shape

        with torch.no_grad():
            if self.current_input is not None and self.restoration_mode == 'direct_injection':
                embeddings = self.original_model.vit.embeddings(self.current_input)
                hidden_original = embeddings
                for i in range(self.target_layer + 1):
                    layer_output = self.original_model.vit.encoder.layer[i](hidden_original)
                    if isinstance(layer_output, tuple):
                        hidden_original = layer_output[0]
                    else:
                        hidden_original = layer_output

                patch_unlearned = hidden_states[:, 1:, :].reshape(-1, hidden_dim)
                patch_original = hidden_original[:, 1:, :].reshape(-1, hidden_dim)

                sae_features_unlearned = self.sae_model.encoder(patch_unlearned - self.sae_model.pre_bias)
                sae_features_original = self.sae_model.encoder(patch_original - self.sae_model.pre_bias)

                sae_features_restored = sae_features_unlearned.clone()
                for expert_idx in self.expert_features:
                    if expert_idx < sae_features_restored.shape[1]:
                        sae_features_restored[:, expert_idx] = sae_features_unlearned[:, expert_idx] + \
                            self.alpha * (sae_features_original[:, expert_idx] - sae_features_unlearned[:, expert_idx])

                restored_patches = self.sae_model.decoder(sae_features_restored) + self.sae_model.pre_bias

                # Reshape back to patch tokens: (batch_size, seq_len-1, hidden_dim)
                restored_patches = restored_patches.view(batch_size, seq_len - 1, hidden_dim)

                cls_unlearned = hidden_states[:, 0:1, :]      # [batch, 1, hidden_dim]
                cls_original = hidden_original[:, 0:1, :]     # [batch, 1, hidden_dim]
                cls_restored = cls_unlearned + self.alpha * (cls_original - cls_unlearned)

                restored_states = hidden_states.clone()

                if self.current_labels is not None and self.target_class is not None:
                    target_mask = (self.current_labels == self.target_class)

                    # Only modify target class samples (CLS + patches)
                    if target_mask.any():
                        restored_states[target_mask, 0:1, :] = cls_restored[target_mask]      # CLS
                        restored_states[target_mask, 1:, :] = restored_patches[target_mask]   # Patches

                else:
                    restored_states[:, 0:1, :] = cls_restored  # CLS
                    restored_states[:, 1:, :] = restored_patches  # Patches

            else:
                flattened = hidden_states.view(-1, hidden_dim)
                sae_features = self.sae_model.encoder(flattened - self.sae_model.pre_bias)

                if self.restoration_mode == 'add':
                    expert_only_features = torch.zeros_like(sae_features)
                    for expert_idx in self.expert_features:
                        if expert_idx < sae_features.shape[1]:
                            expert_only_features[:, expert_idx] = sae_features[:, expert_idx]

                    expert_only_features_amplified = expert_only_features * self.alpha
                    expert_reconstruction = self.sae_model.decoder(expert_only_features_amplified) + self.sae_model.pre_bias
                    expert_reconstruction = expert_reconstruction.view(batch_size, seq_len, hidden_dim)
                    restored_states = hidden_states + expert_reconstruction

                else:
                    restored_states = hidden_states

        if isinstance(output, tuple):
            return (restored_states,) + output[1:]
        else:
            return restored_states

    def remove(self):
        pass
