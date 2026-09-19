"""Minimal pure-Python TFRecord + tf.train.Example reader.

Reads NDWS shards without a TensorFlow dependency. Scope is deliberately
narrow: uncompressed TFRecord files whose payloads are tf.train.Example
protos containing FloatList features — exactly what NDWS ships. Anything
else raises rather than guessing.

TFRecord framing (per the TensorFlow source):
    uint64 length (LE) | uint32 masked_crc32(length) | payload bytes |
    uint32 masked_crc32(payload)

Proto wire format decoded by hand for the fixed message shape:
    Example        { 1: Features }
    Features       { 1: map<string, Feature> } (repeated FeatureEntry)
    FeatureEntry   { 1: key(string), 2: Feature }
    Feature        { 1: BytesList, 2: FloatList, 3: Int64List }
    FloatList      { 1: repeated float (packed) }

CRC32C record checksums are verified when the `crc32c` wheel is available;
otherwise framing consistency (lengths line up over the whole file) is the
integrity check, and raw-file SHA-256 in data/real/MANIFEST.yaml covers
byte identity anyway.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import numpy as np

try:  # optional strict checksums
    import crc32c as _crc32c_mod

    def _masked_crc(data: bytes) -> int:
        crc = _crc32c_mod.crc32c(data)
        return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

except ImportError:  # pragma: no cover - depends on environment
    _masked_crc = None


class TFRecordError(RuntimeError):
    pass


def iter_tfrecord(path: str | Path, verify_crc: bool = True) -> Iterator[bytes]:
    """Yield raw record payloads from an uncompressed TFRecord file."""
    path = Path(path)
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) != 12:
                raise TFRecordError(f"{path}: truncated record header at byte {f.tell()-len(head)}")
            (length,) = struct.unpack("<Q", head[:8])
            (len_crc,) = struct.unpack("<I", head[8:])
            if verify_crc and _masked_crc is not None and _masked_crc(head[:8]) != len_crc:
                raise TFRecordError(f"{path}: length CRC mismatch at byte {f.tell()-12}")
            payload = f.read(length)
            tail = f.read(4)
            if len(payload) != length or len(tail) != 4:
                raise TFRecordError(f"{path}: truncated record payload at byte {f.tell()}")
            if verify_crc and _masked_crc is not None:
                (data_crc,) = struct.unpack("<I", tail)
                if _masked_crc(payload) != data_crc:
                    raise TFRecordError(f"{path}: payload CRC mismatch at byte {f.tell()}")
            yield payload


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise TFRecordError("varint too long")


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    """Yield (field_number, wire_type, value) over a proto message body."""
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field, wire = tag >> 3, tag & 0x7
        if wire == 0:  # varint
            val, pos = _read_varint(buf, pos)
            yield field, wire, val
        elif wire == 2:  # length-delimited
            ln, pos = _read_varint(buf, pos)
            yield field, wire, buf[pos : pos + ln]
            pos += ln
        elif wire == 5:  # 32-bit
            yield field, wire, buf[pos : pos + 4]
            pos += 4
        elif wire == 1:  # 64-bit
            yield field, wire, buf[pos : pos + 8]
            pos += 8
        else:
            raise TFRecordError(f"unsupported wire type {wire} for field {field}")


def parse_example_floats(payload: bytes) -> dict[str, np.ndarray]:
    """Parse a tf.train.Example whose features are all FloatLists."""
    out: dict[str, np.ndarray] = {}
    for f_ex, w_ex, features_buf in _iter_fields(payload):
        if f_ex != 1 or w_ex != 2:
            raise TFRecordError(f"unexpected Example field {f_ex} (wire {w_ex})")
        assert isinstance(features_buf, bytes)
        for f_fe, w_fe, entry_buf in _iter_fields(features_buf):
            if f_fe != 1 or w_fe != 2:
                raise TFRecordError(f"unexpected Features field {f_fe}")
            assert isinstance(entry_buf, bytes)
            key: str | None = None
            values: np.ndarray | None = None
            for f_en, _w_en, val in _iter_fields(entry_buf):
                if f_en == 1:
                    assert isinstance(val, bytes)
                    key = val.decode("utf-8")
                elif f_en == 2:
                    assert isinstance(val, bytes)
                    values = _parse_feature_floatlist(val)
            if key is None or values is None:
                raise TFRecordError("feature map entry missing key or value")
            out[key] = values
    return out


def _parse_feature_floatlist(buf: bytes) -> np.ndarray:
    """Parse a Feature proto, requiring kind == FloatList (field 2)."""
    for field, wire, val in _iter_fields(buf):
        if field == 2 and wire == 2:  # FloatList
            assert isinstance(val, bytes)
            chunks = []
            for f2, w2, v2 in _iter_fields(val):
                if f2 != 1:
                    raise TFRecordError(f"unexpected FloatList field {f2}")
                if w2 == 2:  # packed floats
                    assert isinstance(v2, bytes)
                    chunks.append(np.frombuffer(v2, dtype="<f4"))
                elif w2 == 5:  # single unpacked float
                    assert isinstance(v2, bytes)
                    chunks.append(np.frombuffer(v2, dtype="<f4"))
                else:
                    raise TFRecordError(f"unexpected FloatList wire type {w2}")
            if not chunks:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(chunks)
        if field in (1, 3):
            raise TFRecordError(
                "Feature is a BytesList/Int64List; this reader only supports "
                "FloatList (all NDWS features are FloatLists)"
            )
    raise TFRecordError("Feature proto had no recognized kind")


def iter_examples(path: str | Path) -> Iterator[dict[str, np.ndarray]]:
    """Yield {feature_name: float32 array} per example in a TFRecord shard."""
    for payload in iter_tfrecord(path):
        yield parse_example_floats(payload)
