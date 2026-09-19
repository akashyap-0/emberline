"""Tests for the NDWS spread pipeline (Phase 3)."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from emberline.data.ndws import CHANNEL_CLIP, CHANNELS
from emberline.data.tfrecord_lite import TFRecordError, iter_examples, parse_example_floats
from emberline.spread.model import SpreadUNet, param_count


# ---------------------------------------------------------------------------
# tfrecord_lite: build a tiny Example proto by hand and round-trip it
# ---------------------------------------------------------------------------

def _varint(n: int) -> bytes:
    out = b""
    while True:
        b7 = n & 0x7F
        n >>= 7
        out += bytes([b7 | (0x80 if n else 0)])
        if not n:
            return out


def _field(num: int, payload: bytes) -> bytes:
    return _varint((num << 3) | 2) + _varint(len(payload)) + payload


def _example(features: dict[str, np.ndarray]) -> bytes:
    entries = b""
    for k, v in features.items():
        floats = np.asarray(v, dtype="<f4").tobytes()
        floatlist = _field(1, floats)          # FloatList.value (packed)
        feature = _field(2, floatlist)         # Feature.float_list
        entry = _field(1, k.encode()) + _field(2, feature)
        entries += _field(1, entry)            # Features.feature map entry
    return _field(1, entries)                  # Example.features


def _tfrecord(path, payloads: list[bytes]) -> None:
    with open(path, "wb") as f:
        for p in payloads:
            f.write(struct.pack("<Q", len(p)))
            f.write(b"\0\0\0\0")               # length crc (not verified w/o crc32c)
            f.write(p)
            f.write(b"\0\0\0\0")               # payload crc


def test_tfrecord_lite_roundtrip(tmp_path):
    feats = {"a": np.arange(6, dtype=np.float32), "b": np.array([1.5, -2.5], np.float32)}
    path = tmp_path / "t.tfrecord"
    _tfrecord(path, [_example(feats), _example(feats)])
    got = list(iter_examples(path))
    assert len(got) == 2
    for ex in got:
        assert set(ex) == {"a", "b"}
        np.testing.assert_array_equal(ex["a"], feats["a"])
        np.testing.assert_array_equal(ex["b"], feats["b"])


def test_tfrecord_lite_rejects_truncation(tmp_path):
    path = tmp_path / "bad.tfrecord"
    payload = _example({"a": np.zeros(4, np.float32)})
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(payload) + 50))  # lies about length
        f.write(b"\0\0\0\0")
        f.write(payload)
    with pytest.raises(TFRecordError):
        list(iter_examples(path))


def test_parse_rejects_non_floatlist():
    # Feature carrying an Int64List (field 3) must be refused, not guessed at.
    int64list = _field(1, _varint(7))
    feature = _field(3, int64list)
    entry = _field(1, b"n") + _field(2, feature)
    payload = _field(1, _field(1, entry))
    with pytest.raises(TFRecordError, match="FloatList"):
        parse_example_floats(payload)


# ---------------------------------------------------------------------------
# conversion contract
# ---------------------------------------------------------------------------

def test_channel_order_and_clip_bounds():
    assert CHANNELS[-1] == "PrevFireMask"
    assert len(CHANNELS) == 12
    lo, hi = CHANNEL_CLIP["th"]
    assert (lo, hi) == (0.0, 360.0)  # wind direction is an angle


# ---------------------------------------------------------------------------
# model contract
# ---------------------------------------------------------------------------

def test_unet_under_param_cap_and_shapes():
    m = SpreadUNet(in_channels=12)
    n = param_count(m)
    assert n <= 5_000_000, f"UNet must stay under 5M params, has {n}"
    with torch.no_grad():
        out = m(torch.zeros(2, 12, 64, 64))
    assert out.shape == (2, 64, 64)


def test_masked_loss_excludes_uncertain_pixels():
    # the trainer's masking pattern: -1 target pixels contribute nothing
    logits = torch.full((1, 4, 4), 3.0)
    y = torch.full((1, 4, 4), -1.0)
    y[0, 0, 0] = 1.0
    mask = y != -1
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask], y[mask])
    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([3.0]), torch.tensor([1.0]))
    assert torch.isclose(loss, ref)
