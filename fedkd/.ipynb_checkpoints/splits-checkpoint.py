from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from core import partition, partition_fedkd1

def split_path(data_path, scenario, split_seed, alpha=0.1, partition_name=None):
    root = Path(data_path).resolve().parent / 'splits'
    if partition_name == 'fedkd1':
        return root / f'fedkd1_split{int(split_seed)}.json'
    suffix = f'_alpha{float(alpha):g}' if int(scenario) == 3 else ''
    return root / f'scenario{int(scenario)}_split{int(split_seed)}{suffix}.json'

def build_manifest(data, scenario, split_seed, alpha=0.1, partition=None):
    partition_alpha = float(alpha)
    alpha = partition_alpha if int(scenario) == 3 else None
    n = len(data['private_y'])
    if partition == 'fedkd1':
        clients = partition_fedkd1(data['private_x'], data['private_y'], data['private_device'], seed=int(split_seed))
    else:
        clients = globals()['partition'](data['private_x'], data['private_y'], data['private_device'], scenario=int(scenario), seed=int(split_seed), alpha=partition_alpha)
    private_device = np.asarray(data['private_device'])
    for k, ids in enumerate(clients):
        ids = np.asarray(ids, dtype=int)
        devices = np.unique(private_device[ids]) if len(ids) else np.array([], dtype=int)
        if partition != 'fedkd1' and len(devices) > 1:
            raise ValueError(f'client {k} contains multiple devices: {devices.tolist()}')
    flat = np.concatenate([np.asarray(ids, dtype=int) for ids in clients]) if clients else np.empty(0, dtype=int)
    if len(flat) != n or len(np.unique(flat)) != n or set(flat.tolist()) != set(range(n)):
        raise ValueError('client partition does not cover private samples exactly once')
    client_arrays = [np.asarray(ids, dtype=int) for ids in clients]
    return {'version': 1, 'partition': partition, 'scenario': int(scenario), 'split_seed': int(split_seed), 'alpha': alpha, 'private_indices': list(range(n)), 'open_indices': list(range(len(data['open_y']))), 'test_indices': list(range(len(data['test_y']))), 'private_count': int(n), 'open_count': int(len(data['open_y'])), 'test_count': int(len(data['test_y'])), 'clients': [ids.tolist() for ids in client_arrays], 'client_counts': [int(len(ids)) for ids in client_arrays], 'client_classes': [sorted(set(np.asarray(data['private_y'])[ids].tolist())) for ids in client_arrays], 'client_devices': [sorted(np.unique(private_device[ids]).astype(int).tolist()) if len(ids) else [] for ids in client_arrays], 'empty_clients': [k for k, ids in enumerate(client_arrays) if len(ids) == 0]}

def save_manifest(path, manifest):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

def load_or_create(data, scenario, split_seed, alpha=0.1, path=None, data_path=None, partition=None):
    path = Path(path) if path else split_path(data_path or '../data/mini_nbaiot.npz', scenario, split_seed, alpha, partition)
    expected = build_manifest(data, scenario, split_seed, alpha, partition)
    if path.exists():
        actual = json.loads(path.read_text(encoding='utf-8'))
        keys = ('partition','scenario','split_seed','alpha','private_count','open_count','test_count','clients')
        if any(actual.get(k) != expected.get(k) for k in keys): raise ValueError(f'split manifest mismatch: {path}')
        return [np.asarray(ids, dtype=int) for ids in actual['clients']], path, actual
    save_manifest(path, expected); return [np.asarray(ids, dtype=int) for ids in expected['clients']], path, expected
