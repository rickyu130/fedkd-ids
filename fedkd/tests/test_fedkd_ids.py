import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedkd_ids import (
    FedKDCNN, global_verifier, hard_vote_logits, scenario1_metrics,
)
from fedkd_ids import build_clients, fedkd_round
from core import partition_partition1
from prepare_data import BLOCKS, WINDOWS, build_fedkd_full


def _columns():
    specs = {'MI_dir': ['weight', 'mean', 'variance'], 'H': ['weight', 'mean', 'variance'],
             'HH': ['weight', 'mean', 'std', 'magnitude', 'radius', 'covariance', 'pcc'],
             'HH_jit': ['weight', 'mean', 'variance'],
             'HpHp': ['weight', 'mean', 'std', 'magnitude', 'radius', 'covariance', 'pcc']}
    return [f'{block}_{window}_{stat}' for block, stats in specs.items()
            for window in WINDOWS for stat in stats]


def test_fedkd_cnn_has_paper_detector_and_verifier_heads():
    detector = FedKDCNN(out_features=11)
    verifier = FedKDCNN(out_features=2)
    assert [(m.in_channels, m.out_channels) for m in detector.modules()
            if isinstance(m, nn.Conv1d)] == [(23, 64), (64, 128)]
    assert detector(torch.zeros(3, 23, 5)).shape == (3, 11)
    assert verifier(torch.zeros(3, 23, 5)).shape == (3, 2)


def test_global_verifier_uses_harmonic_mean_threshold_and_filters_outlier():
    # Three collaborators agree and one produces a disagreeing vector.
    logits = np.array([
        [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0],
    ])
    accepted, hm, threshold = global_verifier(logits, threshold_factor=0.5)
    assert accepted.tolist() == [True, True, True, False]
    assert np.all(hm[:3] < threshold)
    assert hm[3] >= threshold


def test_hard_vote_ignores_local_verifier_abstentions():
    logits = np.array([[0, -1, 2, -1], [1, -1, 2, -1], [-1, -1, 1, -1]])
    assert hard_vote_logits(logits).tolist() == [0, -1, 2, -1]


def test_partition1_assigns_each_class_sample_once_with_6_to_10_percent_share():
    y = np.repeat(np.arange(11), 1300)
    population_counts = np.full(11, 1300 / 0.70)
    clients = partition_partition1(y, seed=7, population_counts=population_counts)
    assert len(clients) == 27
    flat = np.concatenate([ids for ids, _labels in clients])
    assert len(flat) == len(y) and len(np.unique(flat)) == len(y)
    for ids, labels in clients:
        support, counts = np.unique(labels, return_counts=True)
        assert 2 <= len(support) <= 7
        for label, count in zip(support, counts):
            fraction = count / population_counts[label]
            assert 0.06 <= fraction <= 0.10


def test_scenario1_metrics_match_attack_positive_binary_definition():
    truth = np.array([0, 0, 1, 2, 0, 3])
    pred = np.array([0, 1, 1, 0, 0, 3])
    metrics = scenario1_metrics(truth, pred)
    assert metrics['accuracy'] == 4 / 6
    assert metrics['precision'] == 2 / 3
    assert metrics['recall'] == 2 / 3
    assert metrics['f1'] == 2 / 3


def test_full_n_baiot_builder_makes_per_class_70_10_20_split(tmp_path):
    names = _columns()
    raw = tmp_path / 'raw'; raw.mkdir()
    for name in ('1.benign.csv', '1.gafgyt.combo.csv', '2.benign.csv', '2.mirai.ack.csv'):
        pd.DataFrame(np.arange(100 * 115, dtype=np.float32).reshape(100, 115), columns=names).to_csv(raw / name, index=False)
    data = build_fedkd_full(raw, seed=11)
    assert len(data['private_y']) == 280
    assert len(data['open_y']) == 40
    assert len(data['test_y']) == 80
    assert data['private_x'].shape == (280, 23, 5)
    assert data['open_x'].shape == (40, 23, 5)
    assert data['test_x'].shape == (80, 23, 5)


def test_fedkd_round_accepts_bounded_client_parallelism():
    rng = np.random.default_rng(4)
    x = rng.random((36, 23, 5), dtype=np.float32)
    y = np.repeat(np.arange(3), 12)
    open_x = rng.random((24, 23, 5), dtype=np.float32)
    clients = build_clients(3, classes=3, seed=17)
    indices = [np.arange(0, 12), np.arange(12, 24), np.arange(24, 36)]
    logits, stats = fedkd_round(
        clients, x, y, indices, open_x, local_epochs=1, verifier_epochs=1,
        distill_epochs=1, batch_size=8, device='cpu', client_workers=2,
    )
    assert logits.shape == (len(open_x),)
    assert stats['uploads'].shape == (3, len(open_x))
    assert all(client.detector_opt is not None for client in clients)
    assert all(client.verifier_opt is not None for client in clients)
