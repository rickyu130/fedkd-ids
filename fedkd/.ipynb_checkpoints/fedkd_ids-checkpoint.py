"""FedKD-IDS detector, local verifier, and server-side logit aggregation."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FedKDCNN(nn.Module):
    """CNN described in FedKD-IDS Table 2 for 23 x 5 N-BaIoT inputs."""

    def __init__(self, out_features=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(23, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 5, 128), nn.ReLU(),
            nn.Linear(128, 115), nn.ReLU(), nn.Linear(115, out_features),
        )

    def forward(self, x):
        return self.net(x)


def predict_logits(model, features, device='cpu', batch_size=8192):
    """Predict deterministically in bounded batches and restore model mode."""
    was_training = model.training
    model.to(device).eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            pieces.append(model(batch).cpu())
    if was_training:
        model.train()
    return torch.cat(pieces).numpy() if pieces else np.empty((0, 0), np.float32)


def train_epochs(model, features, labels, epochs=5, lr=1e-4, batch_size=100,
                 device='cpu', optimizer=None):
    if len(features) == 0:
        return model, optimizer
    model.to(device).train()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(features, dtype=torch.float32),
                      torch.as_tensor(labels, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, pin_memory=str(device).startswith('cuda'),
    )
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    return model, optimizer


def global_verifier(local_logits, threshold_factor=0.5):
    """Hamming-distance/Harmonic-Mean outlier filter (paper Eqs. 17-18).

    Args:
        local_logits: ``(collaborators, public_samples)`` labels; -1 is abstain.
        threshold_factor: tau / mean(HM); Table IV uses 0.5.
    Returns accepted mask, each collaborator's harmonic-mean distance, and tau.
    """
    labels = np.asarray(local_logits, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[0] < 2:
        raise ValueError('global verifier requires at least two collaborator logit rows')
    n_clients, n_samples = labels.shape
    if n_samples == 0:
        raise ValueError('global verifier requires at least one public sample')
    distances = np.zeros((n_clients, n_clients), dtype=np.float64)
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            d = float(np.count_nonzero(labels[i] != labels[j])) / n_samples
            distances[i, j] = distances[j, i] = d
    hm = np.zeros(n_clients, dtype=np.float64)
    for i in range(n_clients):
        row = np.delete(distances[i], i)
        hm[i] = 0.0 if np.any(row == 0) else len(row) / np.reciprocal(row).sum()
    threshold = float(threshold_factor * hm.mean())
    # The paper uses strict HM < tau. In the degenerate all-identical case,
    # retain all collaborators so a zero threshold does not erase all votes.
    accepted = np.ones(n_clients, dtype=bool) if threshold == 0 else hm < threshold
    return accepted, hm, threshold


def hard_vote_logits(local_logits, accepted=None, classes=11):
    labels = np.asarray(local_logits, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError('expected collaborator logits with shape (clients, samples)')
    if accepted is not None:
        accepted = np.asarray(accepted, dtype=bool)
        if accepted.shape != (labels.shape[0],):
            raise ValueError('accepted mask must have one entry per collaborator')
        labels = labels[accepted]
    result = np.full(labels.shape[1], -1, dtype=np.int64)
    if labels.shape[0] == 0:
        return result
    for j in range(labels.shape[1]):
        votes = labels[:, j]
        votes = votes[(votes >= 0) & (votes < classes)]
        if len(votes):
            result[j] = np.bincount(votes, minlength=classes).argmax()
    return result


def scenario1_metrics(y_true, y_pred):
    """Table IV metrics: binary attack-positive precision/recall/F1 + accuracy."""
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.shape != pred.shape:
        raise ValueError('true and predicted labels must have the same shape')
    true_attack = truth != 0
    pred_attack = pred != 0
    tp = int(np.count_nonzero(true_attack & pred_attack))
    fp = int(np.count_nonzero(~true_attack & pred_attack))
    fn = int(np.count_nonzero(true_attack & ~pred_attack))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'accuracy': float(np.mean(truth == pred)), 'precision': precision,
        'recall': recall, 'f1': f1, 'attack_tp': tp, 'attack_fp': fp,
        'attack_fn': fn,
    }


def train_local_verifier(detector, verifier, private_x, open_x, theta=0.8,
                         epochs=5, lr=1e-4, batch_size=100, device='cpu',
                         optimizer=None):
    """Train binary verifier on private-known and confidence-filtered public data."""
    private_verifier_y = np.zeros(len(private_x), dtype=np.int64)
    detector_logits = predict_logits(detector, open_x, device)
    confidence = torch.softmax(torch.as_tensor(detector_logits), dim=1).max(1).values.numpy()
    open_verifier_y = (confidence < theta).astype(np.int64)
    x = np.concatenate((private_x, open_x), axis=0)
    y = np.concatenate((private_verifier_y, open_verifier_y), axis=0)
    return train_epochs(verifier, x, y, epochs, lr, batch_size, device, optimizer)


def collaborator_public_logits(detector, verifier, public_x, device='cpu', batch_size=8192):
    detector_logits = predict_logits(detector, public_x, device, batch_size)
    verifier_logits = predict_logits(verifier, public_x, device, batch_size)
    pred = detector_logits.argmax(1)
    familiar = verifier_logits.argmax(1) == 0
    pred[~familiar] = -1
    return pred


@dataclass
class FedKDClient:
    detector: FedKDCNN
    verifier: FedKDCNN
    detector_opt: object = None
    verifier_opt: object = None


def build_clients(count, classes=11):
    return [FedKDClient(FedKDCNN(classes), FedKDCNN(2)) for _ in range(count)]


def fedkd_round(clients, private_x, private_y, client_indices, public_x,
                local_epochs=5, verifier_epochs=5, distill_epochs=5,
                lr=1e-4, batch_size=100, theta=0.8,
                threshold_factor=0.5, device='cpu', inference_batch=8192):
    """Execute one no-attack FedKD-IDS round and return audit statistics."""
    uploads = []
    for client, ids in zip(clients, client_indices):
        ix = np.asarray(ids, dtype=np.int64)
        client.detector, client.detector_opt = train_epochs(
            client.detector, private_x[ix], private_y[ix], local_epochs,
            lr, batch_size, device, client.detector_opt,
        )
        client.verifier, client.verifier_opt = train_local_verifier(
            client.detector, client.verifier, private_x[ix], public_x, theta,
            verifier_epochs, lr, batch_size, device, client.verifier_opt,
        )
        uploads.append(collaborator_public_logits(
            client.detector, client.verifier, public_x, device, inference_batch,
        ))
    uploaded = np.stack(uploads)
    accepted, hm, threshold = global_verifier(uploaded, threshold_factor)
    global_logits = hard_vote_logits(uploaded, accepted, classes=clients[0].detector.net[-1].out_features)
    valid = global_logits >= 0
    if valid.any():
        for client in clients:
            targets = global_logits[valid]
            client.detector, client.detector_opt = train_epochs(
                client.detector, public_x[valid], targets, distill_epochs,
                lr, batch_size, device, client.detector_opt,
            )
    return global_logits, {
        'accepted': accepted, 'harmonic_mean': hm, 'threshold': threshold,
        'valid_pseudo_labels': int(valid.sum()), 'uploads': uploaded,
    }


def predict_ensemble(clients, features, device='cpu', batch_size=8192):
    predictions = np.stack([
        predict_logits(client.detector, features, device, batch_size).argmax(1)
        for client in clients
    ])
    return hard_vote_logits(predictions, classes=clients[0].detector.net[-1].out_features)
