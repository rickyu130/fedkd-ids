import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from prepare_data import BLOCKS, WINDOWS, build_mini, build_ssfl_ids, build_fedkd_partition1, build_fedkd_full, load_mini


SSFL_IDS_DEVICES = [
    'Danmini_Doorbell', 'Ecobee_Thermostat', 'Philips_B120N10_Baby_Monitor',
    'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
    'SimpleHome_XCS7_1002_WHT_Security_Camera',
    'SimpleHome_XCS7_1003_WHT_Security_Camera', 'Ennio_Doorbell',
    'Samsung_SNH_1011_N_Webcam',
]
SSFL_IDS_CLASSES = ['benign', 'g_combo', 'g_junk', 'g_scan', 'g_tcp', 'g_udp',
                    'm_ack', 'm_scan', 'm_syn', 'm_udp', 'm_udpplain']


def _write_mini_csv(path, n_rows=10):
    columns = [
        f'{block}_{window}_{stat}'
        for window in WINDOWS
        for stat, block in enumerate((BLOCKS[i % len(BLOCKS)] for i in range(23)))
    ]
    data = np.arange(n_rows * len(columns), dtype=float).reshape(n_rows, len(columns))
    pd.DataFrame(data, columns=columns).to_csv(path, index=False)


def _write_ssfl_ids_csv(path, n_rows, offset):
    data = np.arange(offset, offset + n_rows * 115, dtype=float).reshape(n_rows, 115)
    np.savetxt(path, data, delimiter=',')


def _write_ssfl_ids_tree(root, train_rows=10, test_rows=2):
    offset = 0
    for device_idx, device in enumerate(SSFL_IDS_DEVICES):
        device_dir = root / device
        device_dir.mkdir()
        class_names = SSFL_IDS_CLASSES if device_idx < 7 else SSFL_IDS_CLASSES[:6]
        for class_name in class_names:
            _write_ssfl_ids_csv(device_dir / f'{class_name}_train.csv', train_rows, offset)
            offset += train_rows * 115
            _write_ssfl_ids_csv(device_dir / f'{class_name}_test.csv', test_rows, offset)
            offset += test_rows * 115


def test_build_mini_splits_private_open_test(tmp_path):
    _write_mini_csv(tmp_path / '1.benign.csv')

    d = build_mini(tmp_path, per_class=10, seed=7)

    assert d['private_x'].shape == (7, 23, 5)
    assert d['open_x'].shape == (1, 23, 5)
    assert d['test_x'].shape == (2, 23, 5)
    assert d['private_y'].tolist() == [0] * 7


def test_build_mini_per_class_caps_rows(tmp_path):
    _write_mini_csv(tmp_path / '1.benign.csv')

    d = build_mini(tmp_path, per_class=4, seed=7)

    assert sum(len(d[k]) for k in ('private_y', 'open_y', 'test_y')) == 4


def test_load_mini_returns_the_arrays_the_runners_read(tmp_path):
    _write_mini_csv(tmp_path / '1.benign.csv')
    built = build_mini(tmp_path, per_class=10, seed=7)
    path = tmp_path / 'mini.npz'
    np.savez_compressed(path, **{k: v for k, v in built.items() if isinstance(v, np.ndarray)})

    loaded = load_mini(path)

    assert set(loaded) == {'private_x', 'private_y', 'private_device',
                           'open_x', 'open_y', 'test_x', 'test_y'}
    for key, value in loaded.items():
        assert np.array_equal(value, built[key])


def test_build_ssfl_ids_reproduces_device_class_private_open_test_layout(tmp_path):
    _write_ssfl_ids_tree(tmp_path)

    d = build_ssfl_ids(tmp_path, seed=7, private_percent=0.9, open_percent=0.1)

    assert d['private_x'].shape == (801, 23, 5)
    assert d['open_x'].shape == (89, 23, 5)
    assert d['test_x'].shape == (178, 23, 5)
    assert d['private_device'].tolist().count(0) == 99
    assert d['private_device'].tolist().count(7) == 54
    assert sorted(np.unique(d['private_y']).tolist()) == list(range(11))


def test_build_ssfl_ids_has_89_clients_under_ssfl_ids_layout(tmp_path):
    _write_ssfl_ids_tree(tmp_path)
    d = build_ssfl_ids(tmp_path, seed=7, private_percent=0.9, open_percent=0.1)

    from core import partition
    clients = partition(d['private_x'], d['private_y'], d['private_device'],
                         scenario=3, seed=7, alpha=0.1)

    assert len(clients) == 89
    assert sum(len(ids) for ids in clients) == len(d['private_y'])


def test_build_fedkd_partition1_uses_70_10_20_split(tmp_path):
    _write_ssfl_ids_tree(tmp_path, train_rows=8, test_rows=2)

    d = build_fedkd_partition1(tmp_path, seed=7)

    assert d['private_x'].shape == (623, 23, 5)
    assert d['open_x'].shape == (89, 23, 5)
    assert d['test_x'].shape == (178, 23, 5)
    assert len(np.unique(d['private_device'])) == 9
    assert d['private_device'].tolist().count(0) == 77
    assert d['private_device'].tolist().count(7) == 42


def test_build_fedkd_partition1_manifest_has_27_clients(tmp_path):
    _write_ssfl_ids_tree(tmp_path, train_rows=8, test_rows=2)
    d = build_fedkd_partition1(tmp_path, seed=7)

    from splits import build_manifest
    manifest = build_manifest(d, scenario=1, split_seed=7, partition='fedkd1')

    assert manifest['partition'] == 'fedkd1'
    assert len(manifest['clients']) == 27
    assert sum(manifest['client_counts']) == len(d['private_y'])
    assert all(2 <= len(classes) <= 7 for classes in manifest['client_classes'])
