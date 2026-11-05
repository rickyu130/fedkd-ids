"""Prepare the mini-N-BaIoT data used by Zhao et al. (2023).

``build_mini`` draws ``per_class`` rows at random from each raw CSV, matching the
paper's 1000-record-per-device/category construction; ``main`` writes the npz that
the runners read back through ``load_mini``.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LABELS = ['benign','gafgyt.combo','gafgyt.junk','gafgyt.scan','gafgyt.tcp',
          'gafgyt.udp','mirai.ack','mirai.scan','mirai.syn','mirai.udp','mirai.udpplain']

SSFL_IDS_DEVICES = [
    'Danmini_Doorbell', 'Ecobee_Thermostat', 'Philips_B120N10_Baby_Monitor',
    'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
    'SimpleHome_XCS7_1002_WHT_Security_Camera',
    'SimpleHome_XCS7_1003_WHT_Security_Camera', 'Ennio_Doorbell',
    'Samsung_SNH_1011_N_Webcam',
]
SSFL_IDS_CLASSES = ['benign', 'g_combo', 'g_junk', 'g_scan', 'g_tcp', 'g_udp',
                    'm_ack', 'm_scan', 'm_syn', 'm_udp', 'm_udpplain']

WINDOWS = ['L5', 'L3', 'L1', 'L0.1', 'L0.01']
BLOCKS = ['MI_dir', 'H', 'HH', 'HH_jit', 'HpHp']

def window_feature_permutation(feature_names):
    """Return indices that arrange raw N-BaIoT columns by window then block.

    The raw files are block-major (all windows of ``MI_dir``, then ``H``, ...),
    while the paper's 23-by-5 matrix is window-major.  Sorting by the parsed
    window and block preserves each block's statistic order.
    """
    if len(feature_names) != 115:
        raise ValueError(f'expected 115 feature columns, got {len(feature_names)}')
    parsed = []
    for i, name in enumerate(feature_names):
        try:
            block, window, _stat = name.rsplit('_', 2)
            key = (WINDOWS.index(window), BLOCKS.index(block))
        except (AttributeError, ValueError) as exc:
            raise ValueError(f'unrecognized N-BaIoT feature name: {name}') from exc
        parsed.append((key, i))
    perm = [i for _key, i in sorted(parsed)]
    window_counts = np.bincount([key[0] for key, _i in parsed], minlength=len(WINDOWS))
    if not np.all(window_counts == 23):
        raise ValueError('feature columns do not contain 23 features per window')
    return perm

def build_mini(raw_dir='N-BaIoT', per_class=1000, seed=7):

    raw = Path(raw_dir); rows=[]; feature_names=None
    for p in sorted(raw.glob('*.csv')):
        m=re.match(r'(\d+)\.(.+)\.csv$', p.name)
        if not m or m.group(2) not in LABELS: continue
        device=int(m.group(1)); label=LABELS.index(m.group(2))
        # Uniform sampling needs the whole file; the per-file RNG is derived from
        # (seed, device, label) so the draw does not depend on iteration order.
        full=pd.read_csv(p)
        rng=np.random.default_rng([seed,device,label])
        take=np.sort(rng.choice(len(full),size=min(per_class,len(full)),replace=False))
        d=full.iloc[take]
        current_features=list(d.columns[:115])
        if feature_names is None: feature_names=current_features
        elif current_features != feature_names: raise ValueError(f'feature columns differ in {p}')
        d=d.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        d=d.iloc[:, :115]; d['device']=device; d['label']=label; rows.append(d)
    all_d=pd.concat(rows, ignore_index=True)
    rng=np.random.default_rng(seed)
    idx=np.arange(len(all_d)); rng.shuffle(idx); all_d=all_d.iloc[idx].reset_index(drop=True)
    perm=window_feature_permutation(feature_names)
    x=all_d.iloc[:, :115].to_numpy(np.float32)[:, perm]; y=all_d.label.to_numpy(np.int64); device=all_d.device.to_numpy(np.int64)
    lo=x.min(0); hi=x.max(0); x=(x-lo)/(hi-lo+1e-12)
    # Window-major chunks become the paper's 23 rows x 5 columns matrix.
    x=x.reshape(-1,5,23).transpose(0,2,1)
    n=len(y); a=(7*n)//10; b=(8*n)//10
    return {'private_x':x[:a], 'private_y':y[:a], 'private_device':device[:a],
            'open_x':x[a:b], 'open_y':y[a:b], 'test_x':x[b:], 'test_y':y[b:],
            'labels':LABELS, 'feature_min':lo.tolist(), 'feature_max':hi.tolist()}


def _load_ssfl_ids_csv(path):
    values = np.loadtxt(path, delimiter=',', ndmin=2)
    if values.shape[1] != 115:
        raise ValueError(f'{path} must contain 115 feature columns, got {values.shape[1]}')
    # SSFL-IDS-main keeps np.loadtxt's float64 precision until its
    # TensorDataset conversion; preserve that path so the scaler statistics
    # match the reference implementation before the final float32 cast.
    return values.astype(np.float64, copy=False)


def _split_ssfl_ids_train(features, labels, private_percent, open_percent):
    """Match SSFL-IDS-main's per-device SplitPrivateOpen behavior."""
    order = np.arange(len(labels))
    np.random.shuffle(order)
    features = features[order]
    labels = labels[order]
    private_features, private_labels = [], []
    open_features, open_labels = [], []
    for class_id in np.unique(labels):
        ids = np.flatnonzero(labels == class_id)
        private_count = int(len(ids) * private_percent)
        open_count = int(len(ids) * open_percent)
        private_ids = ids[:private_count]
        open_ids = ids[private_count:private_count + open_count]
        private_features.append(features[private_ids])
        private_labels.extend([int(class_id)] * len(private_ids))
        open_features.append(features[open_ids])
        open_labels.extend([int(class_id)] * len(open_ids))
    return (np.concatenate(private_features, axis=0), np.asarray(private_labels, dtype=np.int64),
            np.concatenate(open_features, axis=0), np.asarray(open_labels, dtype=np.int64))


def build_ssfl_ids(raw_dir='SSFL-IDS-main/data/nba_iot_1000', seed=7,
                   private_percent=0.9, open_percent=0.1):
    """Build the exact data layout consumed by SSFL-IDS-main.

    The source tree contains 800 headerless train rows and 200 headerless test
    rows for each device/category file.  Train rows are split per device and
    class into private/open sets, using the same seeded shuffle and percentages
    as SSFL-IDS-main.  Scaling is fitted on the combined open set only, then all
    arrays are reshaped to ``(N, 23, 5)`` without reordering columns.
    """
    if not (0 <= private_percent <= 1 and 0 <= open_percent <= 1):
        raise ValueError('private_percent and open_percent must be between 0 and 1')
    if private_percent + open_percent > 1:
        raise ValueError('private_percent + open_percent must be <= 1')

    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f'SSFL-IDS data directory does not exist: {root}')

    # SSFL-IDS-main seeds NumPy once, then shuffles each device in sequence.
    np.random.seed(seed)
    private_x_parts, private_y_parts, private_device_parts = [], [], []
    open_x_parts, open_y_parts = [], []
    test_x_parts, test_y_parts = [], []
    client_counts = []

    for device_id, device_name in enumerate(SSFL_IDS_DEVICES):
        device_dir = root / device_name
        if not device_dir.is_dir():
            raise FileNotFoundError(f'missing SSFL-IDS device directory: {device_dir}')
        class_names = SSFL_IDS_CLASSES if device_id < 7 else SSFL_IDS_CLASSES[:6]
        train_parts, train_labels = [], []
        test_parts, test_labels = [], []
        for class_id, class_name in enumerate(class_names):
            train_path = device_dir / f'{class_name}_train.csv'
            test_path = device_dir / f'{class_name}_test.csv'
            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(f'missing train/test pair for {device_name}/{class_name}')
            train = _load_ssfl_ids_csv(train_path)
            test = _load_ssfl_ids_csv(test_path)
            train_parts.append(train)
            test_parts.append(test)
            train_labels.extend([class_id] * len(train))
            test_labels.extend([class_id] * len(test))

        train_features = np.concatenate(train_parts, axis=0)
        train_labels = np.asarray(train_labels, dtype=np.int64)
        private, private_labels, open_, open_labels = _split_ssfl_ids_train(
            train_features, train_labels, private_percent, open_percent)
        private_x_parts.append(private)
        private_y_parts.append(private_labels)
        private_device_parts.append(np.full(len(private), device_id, dtype=np.int64))
        open_x_parts.append(open_)
        open_y_parts.append(open_labels)
        test_x_parts.append(np.concatenate(test_parts, axis=0))
        test_y_parts.append(np.asarray(test_labels, dtype=np.int64))
        client_counts.append(len(class_names))

    private_raw = np.concatenate(private_x_parts, axis=0)
    private_y = np.concatenate(private_y_parts, axis=0)
    private_device = np.concatenate(private_device_parts, axis=0)
    open_raw = np.concatenate(open_x_parts, axis=0)
    open_y = np.concatenate(open_y_parts, axis=0)
    test_raw = np.concatenate(test_x_parts, axis=0)
    test_y = np.concatenate(test_y_parts, axis=0)

    # SSFL-IDS-main calls ShuffleDataset on the combined open set before
    # creating clients.  Reproduce that order while keeping labels aligned.
    open_order = np.arange(len(open_y))
    np.random.shuffle(open_order)
    open_raw = open_raw[open_order]
    open_y = open_y[open_order]

    scaler = MinMaxScaler()
    scaler.fit(open_raw)
    private_x = scaler.transform(private_raw).astype(np.float32).reshape(-1, 23, 5)
    open_x = scaler.transform(open_raw).astype(np.float32).reshape(-1, 23, 5)
    test_x = scaler.transform(test_raw).astype(np.float32).reshape(-1, 23, 5)
    return {
        'private_x': private_x, 'private_y': private_y, 'private_device': private_device,
        'open_x': open_x, 'open_y': open_y, 'test_x': test_x, 'test_y': test_y,
        'client_counts': np.asarray(client_counts, dtype=np.int64),
        'labels': np.asarray(LABELS, dtype='<U20'),
        'feature_min': scaler.data_min_.tolist(), 'feature_max': scaler.data_max_.tolist(),
    }


def build_fedkd_partition1(raw_dir='SSFL-IDS-main/data/nba_iot_1000', seed=7):
    """Build the FedKD-IDS Scenario 1, Partition 1 dataset.

    The paper partitions every device/class population into 70% private,
    10% public and 20% test samples.  The repository stores those rows in
    separate ``*_train.csv`` and ``*_test.csv`` files, so this function joins
    them before applying the paper's percentages.
    """
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f'FedKD Partition 1 data directory does not exist: {root}')
    rng = np.random.default_rng(seed)
    private_parts, private_labels, private_devices = [], [], []
    open_parts, open_labels = [], []
    test_parts, test_labels = [], []
    for device_id, device_name in enumerate(SSFL_IDS_DEVICES):
        device_dir = root / device_name
        class_names = SSFL_IDS_CLASSES if device_id < 7 else SSFL_IDS_CLASSES[:6]
        for class_id, class_name in enumerate(class_names):
            train_path = device_dir / f'{class_name}_train.csv'
            test_path = device_dir / f'{class_name}_test.csv'
            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(f'missing train/test pair for {device_name}/{class_name}')
            values = np.concatenate([_load_ssfl_ids_csv(train_path), _load_ssfl_ids_csv(test_path)], axis=0)
            order = rng.permutation(len(values))
            values = values[order]
            private_count = int(len(values) * 0.70)
            open_count = int(len(values) * 0.10)
            private = values[:private_count]
            public = values[private_count:private_count + open_count]
            test = values[private_count + open_count:]
            private_parts.append(private); private_labels.extend([class_id] * len(private))
            private_devices.extend([device_id] * len(private))
            open_parts.append(public); open_labels.extend([class_id] * len(public))
            test_parts.append(test); test_labels.extend([class_id] * len(test))

    private_raw = np.concatenate(private_parts, axis=0)
    open_raw = np.concatenate(open_parts, axis=0)
    test_raw = np.concatenate(test_parts, axis=0)
    open_order = rng.permutation(len(open_raw))
    open_raw = open_raw[open_order]
    open_labels = np.asarray(open_labels, dtype=np.int64)[open_order].tolist()
    scaler = MinMaxScaler().fit(open_raw)
    reshape = lambda a: scaler.transform(a).astype(np.float32).reshape(-1, 23, 5)
    return {
        'private_x': reshape(private_raw),
        'private_y': np.asarray(private_labels, dtype=np.int64),
        'private_device': np.asarray(private_devices, dtype=np.int64),
        'open_x': reshape(open_raw),
        'open_y': np.asarray(open_labels, dtype=np.int64),
        'test_x': reshape(test_raw),
        'test_y': np.asarray(test_labels, dtype=np.int64),
        'labels': np.asarray(LABELS),
        'partition': 'fedkd1',
        'feature_min': scaler.data_min_.tolist(),
        'feature_max': scaler.data_max_.tolist(),
    }


def build_fedkd_full(raw_dir='/root/autodl-tmp/N-BaIoT', seed=7):
    """Load the full, headered N-BaIoT corpus and make the paper's 70/10/20 split.

    Splitting is stratified independently by device and label. The public set is
    used to fit the MinMax scaler, matching the paper's open-data preprocessing.
    """
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f'N-BaIoT directory does not exist: {root}')
    label_to_id = {name: i for i, name in enumerate(LABELS)}
    private_x_parts, private_y_parts, private_device_parts = [], [], []
    open_x_parts, open_y_parts = [], []
    test_x_parts, test_y_parts = [], []
    population_counts = np.zeros(len(LABELS), dtype=np.int64)
    files = []
    for path in root.glob('*.csv'):
        match = re.fullmatch(r'(\d+)\.(.+)\.csv', path.name)
        if match and match.group(2) in label_to_id:
            files.append((int(match.group(1)), label_to_id[match.group(2)], path))
    if not files:
        raise ValueError(f'no N-BaIoT device/category CSV files found under {root}')
    rng = np.random.default_rng(seed)
    for device_id, label, path in sorted(files, key=lambda item: (item[0], item[1])):
        frame = pd.read_csv(path, low_memory=False)
        if frame.shape[1] < 115:
            raise ValueError(f'{path} has {frame.shape[1]} columns; expected at least 115')
        feature_names = list(frame.columns[:115])
        permutation = window_feature_permutation(feature_names)
        frame = frame.iloc[:, :115].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        values = frame.to_numpy(dtype=np.float32, copy=True)[:, permutation]
        count = len(values)
        population_counts[label] += count
        order = rng.permutation(count)
        values = values[order]
        n_private = int(count * 0.70)
        n_open = int(count * 0.10)
        private = values[:n_private]
        public = values[n_private:n_private + n_open]
        test = values[n_private + n_open:]
        private_x_parts.append(private)
        private_y_parts.append(np.full(len(private), label, dtype=np.int64))
        private_device_parts.append(np.full(len(private), device_id, dtype=np.int64))
        open_x_parts.append(public)
        open_y_parts.append(np.full(len(public), label, dtype=np.int64))
        test_x_parts.append(test)
        test_y_parts.append(np.full(len(test), label, dtype=np.int64))

    private_raw = np.concatenate(private_x_parts)
    private_y = np.concatenate(private_y_parts)
    private_device = np.concatenate(private_device_parts)
    open_raw = np.concatenate(open_x_parts)
    open_y = np.concatenate(open_y_parts)
    test_raw = np.concatenate(test_x_parts)
    test_y = np.concatenate(test_y_parts)
    open_order = rng.permutation(len(open_y))
    open_raw, open_y = open_raw[open_order], open_y[open_order]
    scaler = MinMaxScaler().fit(open_raw)
    reshape = lambda array: scaler.transform(array).astype(np.float32).reshape(-1, 23, 5)
    return {
        'private_x': reshape(private_raw), 'private_y': private_y,
        'private_device': private_device, 'open_x': reshape(open_raw),
        'open_y': open_y, 'test_x': reshape(test_raw), 'test_y': test_y,
        'population_counts': population_counts, 'labels': np.asarray(LABELS),
        'partition': 'fedkd1_full', 'feature_min': scaler.data_min_.tolist(),
        'feature_max': scaler.data_max_.tolist(),
    }

def load_mini(path='data/mini_nbaiot.npz'):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--format', choices=['mini', 'ssfl-ids', 'fedkd-partition1', 'fedkd-full'], default='mini')
    ap.add_argument('--raw-dir',default='N-BaIoT')
    ap.add_argument('--ssfl-ids-dir',default='../SSFL-IDS-main/data/nba_iot_1000')
    ap.add_argument('--full-raw-dir',default='/root/autodl-tmp/N-BaIoT')
    ap.add_argument('--out',default='data/mini_nbaiot.npz')
    ap.add_argument('--per-class',type=int,default=1000)
    ap.add_argument('--private-percent',type=float,default=0.9)
    ap.add_argument('--open-percent',type=float,default=0.1)
    ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args()
    if args.format == 'fedkd-full':
        d=build_fedkd_full(args.full_raw_dir,args.seed)
    elif args.format == 'fedkd-partition1':
        d=build_fedkd_partition1(args.ssfl_ids_dir,args.seed)
    elif args.format == 'ssfl-ids':
        d=build_ssfl_ids(args.ssfl_ids_dir,args.seed,args.private_percent,args.open_percent)
    else:
        d=build_mini(args.raw_dir,args.per_class,args.seed)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out, **{k:v for k,v in d.items() if isinstance(v,np.ndarray)})
    out.with_suffix('.json').write_text(json.dumps({
        'format': args.format, 'labels': d['labels'].tolist() if isinstance(d['labels'], np.ndarray) else d['labels'],
        'feature_min': d['feature_min'], 'feature_max': d['feature_max'],
        'population_counts': d.get('population_counts', np.array([])).tolist(),
        'client_counts': d.get('client_counts', []).tolist() if isinstance(d.get('client_counts'), np.ndarray) else d.get('client_counts', []),
    }),encoding='utf-8')
    print(f'wrote {out} ({len(d["private_y"])} private, {len(d["open_y"])} open, {len(d["test_y"])} test)')
if __name__=='__main__': main()
