import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Simple demo for SAE-based unlearning restoration test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test restoration on CIFAR-10 class 0 (airplane)
  python demo.py --dataset cifar10 --unlearned_model my_unlearned_model.pth --target_class 0

  # Test restoration on Imagenette class 5 (French horn) with custom alpha values
  python demo.py --dataset imagenette --unlearned_model my_model.pth --target_class 5 --alpha 1.0 5.0 10.0
        """
    )

    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['cifar10', 'imagenette'],
        help='Dataset name (cifar10 or imagenette)'
    )
    parser.add_argument(
        '--unlearned_model', type=str, required=True,
        help='Path to your unlearned model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--target_class', type=int, required=True,
        help='Target class to test restoration on (0-9)'
    )
    parser.add_argument(
        '--alpha', type=float, nargs='+', default=[1.0, 5.0, 10.0],
        help='Expert feature amplification coefficients (alpha values) to test (default: 1.0 5.0 10.0)'
    )
    parser.add_argument(
        '--layer', type=int, default=9, choices=[8, 9, 10],
        help='ViT layer to apply restoration (default: 9, available: 8, 9, 10)'
    )
    parser.add_argument(
        '--save_dir', type=str, default='./results',
        help='Directory to save results (default: ./results)'
    )

    args = parser.parse_args()

    # Build recovery_test.py command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'recovery_test.py'),
        '--dataset', args.dataset,
        '--unlearned_model', args.unlearned_model,
        '--target_class', str(args.target_class),
        '--layer', str(args.layer),
        '--save_dir', args.save_dir,
        '--alpha'
    ] + [str(a) for a in args.alpha]

    print(f"\nRunning restoration test with the following configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Unlearned model: {args.unlearned_model}")
    print(f"  Target class: {args.target_class}")
    print(f"  Layer: {args.layer}")
    print(f"  Alpha values (α): {args.alpha}")
    print(f"  Save directory: {args.save_dir}")
    print()

    # Run restoration test
    import subprocess
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
