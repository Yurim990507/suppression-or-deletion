import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress transformers warnings about weight initialization
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

from models import get_model, set_seed
from datasets import get_dataset, get_dataloader, get_class_names
from utils import load_sae_model, load_activation_stats, SAERestorationHook, evaluate_model_performance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test SAE-based restoration on unlearned models"
    )

    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['cifar10', 'imagenette'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_path', type=str, default='./data',
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--unlearned_model', type=str, required=True,
        help='Path to unlearned model checkpoint'
    )
    parser.add_argument(
        '--original_model', type=str, default=None,
        help='Path to original model checkpoint (auto-detected if not provided)'
    )

    parser.add_argument(
        '--sae_model', type=str, default=None,
        help='Path to SAE model (auto-detected if not provided)'
    )
    parser.add_argument(
        '--expert_features', type=str, default=None,
        help='Path to expert features file (auto-detected if not provided)'
    )
    parser.add_argument(
        '--layer', type=int, default=9, choices=[8, 9, 10],
        help='ViT layer to apply restoration (default: 9, available: 8, 9, 10)'
    )

    parser.add_argument(
        '--target_class', type=int, required=True,
        help='Target class to restore'
    )
    parser.add_argument(
        '--alpha', type=float, nargs='+', default=[1.0, 5.0, 10.0],
        help='Expert feature amplification coefficients (alpha values) to test'
    )
    parser.add_argument(
        '--restoration_mode', type=str, default='direct_injection',
        choices=['add', 'direct_injection'],
        help='Restoration mode (default: direct_injection with gradual addition)'
    )

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')

    return parser.parse_args()


def get_dataset_config(dataset_name):
    configs = {
        'cifar10': {'k': 16},
        'imagenette': {'k': 32},
    }
    return configs.get(dataset_name.lower(), {'k': 16})


def auto_detect_paths(args):
    pretrained_dir = os.path.join(os.path.dirname(__file__), 'pretrained', args.dataset)

    dataset_config = get_dataset_config(args.dataset)
    k_value = dataset_config['k']

    if args.original_model is None:
        original_path = os.path.join(pretrained_dir, 'vit_base_16_original.pth')
        if os.path.exists(original_path):
            args.original_model = original_path
            print(f"Auto-detected original model: {original_path}")
        else:
            raise FileNotFoundError(
                f"Original model not found at {original_path}. "
                "Please specify --original_model manually."
            )

    if args.sae_model is None:
        sae_path = os.path.join(pretrained_dir, f'sae_layer{args.layer}_k{k_value}.pt')
        if os.path.exists(sae_path):
            args.sae_model = sae_path
            print(f"Auto-detected SAE model: {sae_path}")
        else:
            raise FileNotFoundError(
                f"SAE model not found at {sae_path}. "
                "Please specify --sae_model manually."
            )

    stats_path = os.path.join(pretrained_dir, f'activations_layer{args.layer}_stats.npy')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Activation stats not found at {stats_path}")
    args.stats_path = stats_path

    if args.expert_features is None:
        expert_path = os.path.join(pretrained_dir, f'expert_features_layer{args.layer}_k{k_value}.pt')
        if os.path.exists(expert_path):
            args.expert_features = expert_path
            print(f"Auto-detected expert features: {expert_path}")
        else:
            raise FileNotFoundError(
                f"Expert features not found at {expert_path}. "
                "Please specify --expert_features manually."
            )


def load_expert_features(expert_path, target_class):
    expert_data = torch.load(expert_path, weights_only=False)

    if 'class_experts_details' in expert_data:
        class_experts = expert_data['class_experts_details']
    elif isinstance(expert_data, dict) and target_class in expert_data:
        class_experts = expert_data
    else:
        raise ValueError(f"Unexpected expert features format in {expert_path}")

    if target_class not in class_experts:
        raise ValueError(f"Target class {target_class} not found in expert features")

    expert_list = class_experts[target_class]

    if isinstance(expert_list, list) and len(expert_list) > 0:
        if isinstance(expert_list[0], dict):
            expert_features = [f['feature_id'] for f in expert_list]
        else:
            expert_features = expert_list
    else:
        expert_features = expert_list

    return expert_features


def create_visualization(results, target_class, class_names, save_path):
    alphas = sorted(results.keys())
    target_accs = [results[a]['target_acc'] for a in alphas]
    retain_accs = [results[a]['retain_acc'] for a in alphas]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(alphas, target_accs, marker='o', linewidth=2.5, markersize=8,
            color='red', label=f'Target: {class_names[target_class]}')
    ax.plot(alphas, retain_accs, marker='s', linewidth=2.5, markersize=8,
            color='gray', label='Retain classes (avg)', alpha=0.7)

    ax.set_xlabel('Amplification Coefficient (α)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'SAE Restoration Test - Class {target_class} ({class_names[target_class]})',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved: {save_path}")


def main():
    args = parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    auto_detect_paths(args)

    num_classes = 10  
    class_names = get_class_names(args.dataset)

    dataset_config = get_dataset_config(args.dataset)

    print(f"\n{'='*80}")
    print(f"SAE Restoration Test")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"SAE Configuration: k={dataset_config['k']}")
    print(f"Target Class: {args.target_class} ({class_names[args.target_class]})")
    print(f"Layer: {args.layer}")
    print(f"Amplification coefficients (α): {args.alpha}")
    print(f"Restoration Mode: {args.restoration_mode}")
    print(f"{'='*80}\n")

    print("Loading dataset...")
    test_dataset = get_dataset(args.dataset, args.data_path, train=False)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    print("Loading original model...")
    original_model = get_model('vit_base_16', num_classes).to(device)
    original_checkpoint = torch.load(args.original_model, map_location=device, weights_only=False)
    original_model.load_state_dict(original_checkpoint['model_state_dict'])
    original_model.eval()

    print("Loading unlearned model...")
    unlearned_model = get_model('vit_base_16', num_classes).to(device)
    unlearned_checkpoint = torch.load(args.unlearned_model, map_location=device, weights_only=False)

    if 'model_state_dict' in unlearned_checkpoint:
        unlearned_model.load_state_dict(unlearned_checkpoint['model_state_dict'])
    else:
        unlearned_model.load_state_dict(unlearned_checkpoint)

    unlearned_model.eval()

    print("Evaluating unlearned model...")
    unlearned_overall, unlearned_classes = evaluate_model_performance(
        unlearned_model, test_loader, device, num_classes
    )

    retain_classes = [unlearned_classes[i] for i in range(num_classes) if i != args.target_class]
    retain_avg = np.mean(retain_classes) if retain_classes else 0.0

    print(f"\nUnlearned Model Performance:")
    print(f"  Overall: {unlearned_overall:.4f}")
    print(f"  Target class {args.target_class}: {unlearned_classes[args.target_class]:.4f}")
    print(f"  Retain classes (avg): {retain_avg:.4f}")

    print("\nLoading SAE...")
    sae_model, sae_config = load_sae_model(args.sae_model, device)
    activation_stats = load_activation_stats(args.stats_path)

    print("Loading expert features...")
    expert_features = load_expert_features(args.expert_features, args.target_class)
    print(f"Loaded {len(expert_features)} expert features for class {args.target_class}")

    results = {}

    for alpha_val in args.alpha:
        print(f"\n--- Testing α = {alpha_val} ---")

        restoration_hook = SAERestorationHook(
            sae_model, args.layer, activation_stats, expert_features,
            original_model, device, alpha=alpha_val,
            restoration_mode=args.restoration_mode, target_class=args.target_class
        )

        target_layer_path = f"vit.encoder.layer.{args.layer}.output"
        target_module = unlearned_model.get_submodule(target_layer_path)
        handle = target_module.register_forward_hook(restoration_hook)

        restored_overall, restored_classes = evaluate_model_performance(
            unlearned_model, test_loader, device, num_classes, restoration_hook=restoration_hook
        )

        restored_retain = [restored_classes[i] for i in range(num_classes) if i != args.target_class]
        restored_retain_avg = np.mean(restored_retain) if restored_retain else 0.0

        print(f"\nRestored Model Performance (α = {alpha_val}):")
        print(f"  Overall: {restored_overall:.4f}")
        print(f"  Target class {args.target_class}: {restored_classes[args.target_class]:.4f} "
              f"(Δ {restored_classes[args.target_class] - unlearned_classes[args.target_class]:+.4f})")
        print(f"  Retain classes (avg): {restored_retain_avg:.4f}")

        handle.remove()
        restoration_hook.remove()

        results[alpha_val] = {
            'target_acc': restored_classes[args.target_class],
            'retain_acc': restored_retain_avg,
            'overall_acc': restored_overall,
        }

    print("\nCreating visualization...")
    vis_path = os.path.join(args.save_dir, f"restoration_class{args.target_class}.png")
    create_visualization(results, args.target_class, class_names, vis_path)

    results_summary = {
        'dataset': args.dataset,
        'sae_config': {
            'k': dataset_config['k'],
            'layer': args.layer,
        },
        'target_class': args.target_class,
        'class_name': class_names[args.target_class],
        'num_expert_features': len(expert_features),
        'alpha_values': args.alpha,
        'unlearned_performance': {
            'overall': float(unlearned_overall),
            'target': float(unlearned_classes[args.target_class]),
            'retain_avg': float(retain_avg),
        },
        'restoration_results': {
            str(a): {k: float(v) for k, v in res.items()}
            for a, res in results.items()
        }
    }

    import json
    results_path = os.path.join(args.save_dir, f"restoration_class{args.target_class}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved: {results_path}")
    print("\n" + "="*80)
    print("Restoration test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
