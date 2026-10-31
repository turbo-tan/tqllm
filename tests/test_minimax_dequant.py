import importlib.util
import pathlib

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "minimax_dequant", pathlib.Path(__file__).resolve().parents[1] / "tools" / "minimax_dequant.py")
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)


def test_e4m3_known_codes():
    # 0x38 = 1.0, 0x7e = 448 (max), 0x01 = smallest subnormal 2^-9, 0xc0 = -2.0, 0x7f = NaN
    v = md.e4m3_to_f32(np.array([0x38, 0x7E, 0x01, 0xC0, 0x00], np.uint8))
    assert v.tolist() == [1.0, 448.0, 2.0 ** -9, -2.0, 0.0]
    assert np.isnan(md.e4m3_to_f32(np.array([0x7F], np.uint8)))[0]


def test_e4m3_matches_torch():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8")
    b = torch.arange(256, dtype=torch.uint8)
    ref = b.view(torch.float8_e4m3fn).float().numpy()
    ours = md.e4m3_to_f32(b.numpy())
    assert np.array_equal(np.isnan(ref), np.isnan(ours))
    assert np.array_equal(ref[~np.isnan(ref)], ours[~np.isnan(ours)])


def test_e8m0():
    assert md.e8m0_to_f32(np.array([127, 128, 120], np.uint8)).tolist() == [1.0, 2.0, 2.0 ** -7]
    assert np.isnan(md.e8m0_to_f32(np.array([255], np.uint8)))[0]


def test_e2m1_low_nibble_first():
    # byte 0x72: low nibble 2 -> 1.0 (element 0), high nibble 7 -> 6.0 (element 1); 0x9f: 0xf -> -6, 0x9 -> -0.5
    assert md.e2m1_unpack(np.array([[0x72, 0x9F]], np.uint8)).tolist() == [[1.0, 6.0, -6.0, -0.5]]


def test_mxfp8_block_scales_apply_per_32():
    w = np.full((2, 64), 0x38, np.uint8)  # all 1.0
    s = np.array([[127, 128], [126, 129]], np.uint8)
    out = md.dequant_mxfp8(w, s)
    assert out[0, :32].tolist() == [1.0] * 32 and out[0, 32:].tolist() == [2.0] * 32
    assert out[1, :32].tolist() == [0.5] * 32 and out[1, 32:].tolist() == [4.0] * 32
    with pytest.raises(ValueError):
        md.dequant_mxfp8(w, s[:, :1])


def test_nvfp4_group_scale_over_global():
    packed = np.full((1, 16), 0x22, np.uint8)  # 32 values of 1.0
    scale = np.array([[0x38, 0x40]], np.uint8)  # 1.0, 2.0 per group of 16
    out = md.dequant_nvfp4(packed, scale, 4.0)
    assert out[0, :16].tolist() == [0.25] * 16 and out[0, 16:].tolist() == [0.5] * 16
