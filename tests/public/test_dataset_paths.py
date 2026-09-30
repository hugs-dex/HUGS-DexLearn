"""Dataset references must survive relocation without changing numeric fields."""
from pathlib import Path

import numpy as np
import pytest

from dexlearn.utils.resources import portable_dataset_record, resolve_dataset_path


def test_record_round_trip_after_relocation(tmp_path, monkeypatch):
    first, second = tmp_path / 'first', tmp_path / 'second'
    rel = 'object/DGN_2k/scene_cfg/example.npy'
    source = {'scene_path': str(first / rel), 'pose': np.arange(7), 'nested': {'pc_path': str(first / 'points.npy')}}
    portable = portable_dataset_record(source, first)
    assert portable['scene_path'] == rel
    assert portable['pose'] is source['pose']
    assert source['scene_path'] == str(first / rel)
    monkeypatch.setenv('HUGS_DATASET_ROOT', str(second))
    assert resolve_dataset_path(portable['scene_path']) == str(second / rel)
    assert resolve_dataset_path(portable['nested']['pc_path']) == str(second / 'points.npy')


def test_bundle_symlink_keeps_public_layout(tmp_path):
    actual = tmp_path / 'storage'
    actual.mkdir()
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'object').symlink_to(actual, target_is_directory=True)
    assert portable_dataset_record({'scene_path': str(bundle / 'object/a.npy')}, bundle)['scene_path'] == 'object/a.npy'


def test_dataset_reference_cannot_escape_root():
    with pytest.raises(ValueError, match='escapes'):
        resolve_dataset_path('../other/scene.npy')


def test_numpy_path_arrays_preserve_shape(tmp_path):
    record = {'scene_path': np.array([[str(tmp_path / 'scene.npy')]]),
              'scalar': np.asarray(str(tmp_path / 'points.npy'))}
    saved = portable_dataset_record(record, tmp_path)
    assert saved['scene_path'].shape == (1, 1)
    assert saved['scene_path'][0, 0] == 'scene.npy'
    assert saved['scalar'].shape == ()
    assert saved['scalar'].item() == 'points.npy'
