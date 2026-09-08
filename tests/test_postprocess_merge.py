"""Unit tests for tme.scripts.postprocess.normalize_input."""

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pytest

from tme.utils import serialization
from tme.scripts.postprocess import normalize_input


def _make_pickle(
    tmp_path: Path,
    name: str,
    scores: np.ndarray,
    rotations: np.ndarray,
    rotation_mapping: dict,
    variance: float = None,
) -> str:
    """Write a postprocess-compatible pickle to tmp_path/name.pickle."""
    path = str(tmp_path / f"{name}.pickle")
    offset = np.zeros(scores.ndim, dtype=int)

    cli_args = argparse.Namespace(
        template="dummy.mrc",
        target="dummy.mrc",
        centering=True,
        template_mask=None,
        batch_dims=(),
    )
    metadata = (
        np.zeros(scores.ndim),
        np.zeros(scores.ndim),
        np.ones(scores.ndim) * 5.0,
        cli_args,
    )

    if variance is None:
        data = (
            scores.astype(np.float32),
            offset,
            rotations.astype(np.int32),
            rotation_mapping,
            metadata,
        )
    else:
        data = (
            scores.astype(np.float32),
            offset,
            rotations.astype(np.int32),
            rotation_mapping,
            np.float32(variance),
            metadata,
        )
    serialization.serialize(data, path)
    return path


def _identity_rotmat():
    return np.eye(3)


def _basic_inputs(tmp_path: Path):
    """Two foregrounds with identical shape, different score peaks."""
    shape = (4, 4, 4)
    scores_a = np.zeros(shape, dtype=np.float32)
    scores_a[1, 1, 1] = 0.8
    scores_a[2, 2, 2] = 0.5
    rotations_a = np.full(shape, -1, dtype=np.int32)
    rotations_a[1, 1, 1] = 0
    rotations_a[2, 2, 2] = 0

    scores_b = np.zeros(shape, dtype=np.float32)
    scores_b[1, 1, 1] = 0.4
    scores_b[3, 3, 3] = 0.9
    rotations_b = np.full(shape, -1, dtype=np.int32)
    rotations_b[1, 1, 1] = 0
    rotations_b[3, 3, 3] = 0

    rmap = {0: _identity_rotmat()}
    path_a = _make_pickle(tmp_path, "fg_a", scores_a, rotations_a, rmap)
    path_b = _make_pickle(tmp_path, "fg_b", scores_b, rotations_b, rmap)
    return [path_a, path_b], shape


def test_two_foregrounds_smoke(tmp_path):
    foregrounds, shape = _basic_inputs(tmp_path)
    data, entities = normalize_input(
        foregrounds=tuple(foregrounds),
        backgrounds=(),
        compute_snr=False,
        compute_stats=False,
    )
    scores_out = data[0]
    assert scores_out.shape == shape
    assert scores_out[1, 1, 1] == pytest.approx(0.8)
    assert scores_out[3, 3, 3] == pytest.approx(0.9)
    assert scores_out[2, 2, 2] == pytest.approx(0.5)


def test_foreground_shape_mismatch_raises(tmp_path):
    foregrounds, _ = _basic_inputs(tmp_path)

    scores_c = np.zeros((4, 4, 5), dtype=np.float32)
    rotations_c = np.full((4, 4, 5), -1, dtype=np.int32)
    foregrounds.append(
        _make_pickle(tmp_path, "fg_c", scores_c, rotations_c, {0: _identity_rotmat()})
    )

    with pytest.raises(ValueError, match="fg_c"):
        normalize_input(
            foregrounds=tuple(foregrounds),
            backgrounds=(),
            compute_snr=False,
            compute_stats=False,
        )


def test_background_shape_mismatch_raises(tmp_path):
    foregrounds, _ = _basic_inputs(tmp_path)

    scores_bg = np.zeros((4, 4, 5), dtype=np.float32)
    rotations_bg = np.full((4, 4, 5), -1, dtype=np.int32)
    background = _make_pickle(
        tmp_path, "bg", scores_bg, rotations_bg, {0: _identity_rotmat()}
    )

    with pytest.raises(ValueError, match="bg"):
        normalize_input(
            foregrounds=tuple(foregrounds),
            backgrounds=(background,),
            compute_snr=False,
            compute_stats=False,
        )

def test_n_false_positives_uses_max_variance(tmp_path):
    shape = (4, 4, 4)
    scores_a = np.zeros(shape, dtype=np.float32)
    scores_b = np.zeros(shape, dtype=np.float32)
    rotations = np.full(shape, -1, dtype=np.int32)
    rmap = {0: _identity_rotmat()}

    path_a = _make_pickle(tmp_path, "fg_a", scores_a, rotations, rmap, variance=0.5)
    path_b = _make_pickle(tmp_path, "fg_b", scores_b, rotations, rmap, variance=2.0)

    data, _ = normalize_input(
        foregrounds=(path_a, path_b),
        backgrounds=(),
        compute_snr=False,
        compute_stats=False,
    )

    assert len(data) == 6
    assert float(data[4]) == pytest.approx(2.0)


def test_no_stats_skips_entities(tmp_path):
    foregrounds, shape = _basic_inputs(tmp_path)
    data, entities = normalize_input(
        foregrounds=tuple(foregrounds),
        backgrounds=(),
        compute_snr=False,
        compute_stats=False,
    )
    assert entities is None
    assert data[0].shape == shape


def _redirect_tmpdir(monkeypatch, path):
    """Point both $TMPDIR and tempfile's cached default at `path`."""
    import tempfile

    monkeypatch.setenv("TMPDIR", str(path))
    monkeypatch.setattr(tempfile, "tempdir", str(path))


def test_memmap_scratch_matches_ram(tmp_path, monkeypatch):
    foregrounds, _ = _basic_inputs(tmp_path)

    ram_data, _ = normalize_input(
        foregrounds=tuple(foregrounds),
        backgrounds=(),
        compute_snr=False,
        compute_stats=False,
    )

    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    _redirect_tmpdir(monkeypatch, scratch_dir)
    mm_data, _ = normalize_input(
        foregrounds=tuple(foregrounds),
        backgrounds=(),
        compute_snr=False,
        compute_stats=False,
        use_memmap=True,
    )
    np.testing.assert_array_equal(np.asarray(ram_data[0]), np.asarray(mm_data[0]))
    np.testing.assert_array_equal(np.asarray(ram_data[2]), np.asarray(mm_data[2]))


def test_memmap_scratch_cleanup(tmp_path, monkeypatch):
    foregrounds, _ = _basic_inputs(tmp_path)
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    _redirect_tmpdir(monkeypatch, scratch_dir)

    from tme.scripts.postprocess import _cleanup_scratch_files, _SCRATCH_FILES

    normalize_input(
        foregrounds=tuple(foregrounds),
        backgrounds=(),
        compute_snr=False,
        compute_stats=False,
        use_memmap=True,
    )

    _cleanup_scratch_files()
    leftover = sorted(glob(str(scratch_dir / "*.mm")))
    assert leftover == []
    assert _SCRATCH_FILES == []
