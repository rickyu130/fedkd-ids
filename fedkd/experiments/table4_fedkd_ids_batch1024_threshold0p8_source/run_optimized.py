"""GPU-optimized FedKD-IDS runner.

The optimized path keeps each client's current training tensor on the selected
device, avoiding DataLoader batch transfers. It preserves the same model,
epochs, learning rate, threshold and no-checkpoint behavior as ``run.py``.
"""
from run import main, parser


if __name__ == '__main__':
    args = parser().parse_args()
    args.fast = True
    main(args)
