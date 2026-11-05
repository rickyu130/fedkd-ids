"""Run the FedKD-IDS Scenario 1 / Table IV experiment."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from core import partition_partition1
from fedkd_ids import build_clients, fedkd_round, predict_ensemble, scenario1_metrics
from prepare_data import build_fedkd_full, load_mini


def extended_metrics(y_true, y_pred, classes=11):
    """Legacy macro diagnostic retained for the existing SSFL tests/tools."""
    labels = list(range(classes))
    values = {
        'accuracy': float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        'macro_precision': float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
    }
    return values, confusion_matrix(y_true, y_pred, labels=labels)


def _set_seed(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    _set_seed(args.seed, args.deterministic)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'{args.device} requested but CUDA is unavailable')
    data_path = Path(args.data)
    if args.build_data or not data_path.exists():
        data = build_fedkd_full(args.raw_dir, args.seed)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(data_path, **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
        data_path.with_suffix('.json').write_text(json.dumps({
            'format': 'fedkd-full', 'seed': args.seed,
            'population_counts': data['population_counts'].tolist(),
            'feature_min': data['feature_min'], 'feature_max': data['feature_max'],
        }, indent=2), encoding='utf-8')
    else:
        data = load_mini(data_path)
    required = {'private_x', 'private_y', 'open_x', 'test_x', 'test_y', 'population_counts'}
    missing = required - set(data)
    if missing:
        raise ValueError(f'data file is missing FedKD full-data fields: {sorted(missing)}')

    split = partition_partition1(
        data['private_y'], seed=args.split_seed,
        population_counts=data['population_counts'],
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        'method': 'fedkd-ids', 'scenario': 1, 'partition': 1,
        'seed': args.seed, 'split_seed': args.split_seed,
        'private_count': int(len(data['private_y'])),
        'public_count': int(len(data['open_y'])),
        'test_count': int(len(data['test_y'])),
        'collaborators': [{
            'id': i, 'sample_count': int(len(ids)),
            'class_counts': {str(label): int(np.count_nonzero(local_y == label))
                             for label in np.unique(local_y)},
        } for i, (ids, local_y) in enumerate(split)],
    }
    manifest_path = output.with_name(output.stem + '_partition1_manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Indices are retained separately from JSON to keep the manifest compact.
    np.savez_compressed(output.with_name(output.stem + '_partition1_indices.npz'),
                        **{f'client_{i}': ids for i, (ids, _labels) in enumerate(split)})

    clients = build_clients(len(split), classes=11, seed=args.seed)
    config = {
        'method': 'FedKD-IDS', 'scenario': 1, 'partition': 1,
        'threshold': '0.5 * mean(HM)', 'local_verifier_theta': args.theta,
        'rounds': args.rounds, 'local_epochs': args.local_epochs,
        'verifier_epochs': args.verifier_epochs, 'distill_epochs': args.distill_epochs,
        'batch_size': args.batch, 'learning_rate': args.lr,
        'client_workers': args.client_workers,
        'threshold_factor': args.threshold_factor, 'seed': args.seed,
        'split_seed': args.split_seed, 'device': args.device,
        'data': str(data_path.resolve()),
        'collaborators': len(split),
    }
    output.with_suffix('.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    fields = ['round', 'accepted_collaborators', 'threshold', 'mean_hm',
              'valid_pseudo_labels', 'round_seconds', 'accuracy', 'precision', 'recall', 'f1']
    with output.open('w', newline='', encoding='utf-8') as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()

    final_stats = None
    for rnd in range(args.rounds):
        started = time.perf_counter()
        global_labels, stats = fedkd_round(
            clients, data['private_x'], data['private_y'], [ids for ids, _ in split],
            data['open_x'], local_epochs=args.local_epochs,
            verifier_epochs=args.verifier_epochs, distill_epochs=args.distill_epochs,
            lr=args.lr, batch_size=args.batch, theta=args.theta,
            threshold_factor=args.threshold_factor, device=args.device,
            inference_batch=args.inference_batch, client_workers=args.client_workers,
            fast=args.fast,
        )
        row = {
            'round': rnd + 1, 'accepted_collaborators': int(stats['accepted'].sum()),
            'threshold': stats['threshold'], 'mean_hm': float(stats['harmonic_mean'].mean()),
            'valid_pseudo_labels': stats['valid_pseudo_labels'],
            'round_seconds': time.perf_counter() - started,
            'accuracy': '', 'precision': '', 'recall': '', 'f1': '',
        }
        final_stats = stats
        if (rnd + 1) % args.evaluate_every == 0 or rnd + 1 == args.rounds:
            prediction = predict_ensemble(clients, data['test_x'], args.device, args.inference_batch)
            metrics = scenario1_metrics(data['test_y'], prediction)
            row.update({key: metrics[key] for key in ('accuracy', 'precision', 'recall', 'f1')})
            np.savetxt(output.with_name(output.stem + '_confusion.csv'),
                       confusion_matrix(data['test_y'], prediction, labels=np.arange(11)),
                       fmt='%d', delimiter=',')
        with output.open('a', newline='', encoding='utf-8') as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
        print(
            f"fedkd-ids round={rnd + 1}/{args.rounds} accepted={row['accepted_collaborators']}/{len(clients)} "
            f"pseudo={row['valid_pseudo_labels']} hm_mean={row['mean_hm']:.6f} "
            f"accuracy={row['accuracy'] or 'n/a'} f1={row['f1'] or 'n/a'}",
            flush=True,
        )
    if final_stats is not None:
        np.savez_compressed(output.with_name(output.stem + '_global_verifier_last_round.npz'),
                            accepted=final_stats['accepted'], hm=final_stats['harmonic_mean'],
                            threshold=final_stats['threshold'], uploads=final_stats['uploads'])


def parser():
    p = argparse.ArgumentParser(description='FedKD-IDS Scenario 1 / Partition 1')
    p.add_argument('--data', default='../data/nba_iot_fedkd_full.npz')
    p.add_argument('--raw-dir', default='/root/autodl-tmp/N-BaIoT')
    p.add_argument('--build-data', action='store_true')
    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--local-epochs', type=int, default=5)
    p.add_argument('--verifier-epochs', type=int, default=5)
    p.add_argument('--distill-epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--theta', type=float, default=0.8)
    p.add_argument('--threshold-factor', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--split-seed', type=int, default=7)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--inference-batch', type=int, default=8192)
    p.add_argument('--client-workers', type=int, default=1,
                   help='parallel client workers; 1 preserves sequential execution')
    p.add_argument('--fast', action='store_true',
                   help='keep each training stage on the selected device')
    p.add_argument('--evaluate-every', type=int, default=10)
    p.add_argument('--out', default='results/table4_fedkd_ids_0p5mean.csv')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
