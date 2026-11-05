# FedKD-IDS reproduction

`fedkd/run.py` runs FedKD-IDS (Quyen et al., Information Fusion 117, 2025),
Scenario 1 / Partition 1. It is independent of the SSFL implementation in
`SSFL-IDS-main` and includes the paper's detector/local verifier pair, server
Hamming-distance and Harmonic-Mean filtering, and hard-vote distillation.

The full N-BaIoT loader reads headered `device.label.csv` files and makes a
stratified per-device/per-label split of 70% private, 10% public and 20% test.
Partition 1 creates 27 collaborators. Each private label's samples are spread
across eight collaborators, representing 8.75% of that label's full population
per collaborator and giving each collaborator 2–7 labels. Public data fits the
MinMax scaler. The paper does not publish its exact collaborator manifest, so
the generated manifest and sample-index archive are saved beside the results.

On the server with the full data at `/root/autodl-tmp/N-BaIoT`:

```bash
cd /root/FSSL/fedkd
/root/miniconda3/bin/python run.py \
  --data /root/autodl-tmp/fedkd/table4_partition1_full.npz \
  --raw-dir /root/autodl-tmp/N-BaIoT \
  --build-data \
  --rounds 100 \
  --local-epochs 5 \
  --verifier-epochs 5 \
  --distill-epochs 5 \
  --batch 100 \
  --lr 1e-4 \
  --theta 0.8 \
  --threshold-factor 0.5 \
  --device cuda \
  --out results/table4_fedkd_ids_0p5mean.csv
```

The Table IV metrics use the paper's binary definition with all attack classes
combined as the positive class. Accuracy and the saved 11-class confusion
matrix retain the original N-BaIoT labels. The final row reports Accuracy,
Precision, Recall and F1 for comparison with the paper's Partition 1 target:
85.24%, 82.86%, 89.09% and 85.86%, respectively.
