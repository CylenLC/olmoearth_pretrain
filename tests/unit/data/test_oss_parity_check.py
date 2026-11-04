"""Ensure the read-only parity diagnostic rejects real differences."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("s3fs")


@pytest.fixture
def parity_module(monkeypatch):
    """Load the source-tree diagnostic even when pytest is a console script."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    from scripts.diagnostics import check_oss_dataset

    return check_oss_dataset


def test_exact_nan_and_inf(parity_module) -> None:
    """Equal nonfinite positions are accepted for arrays, scalars and tensors."""
    array = np.array([1, np.nan, np.inf, -np.inf], dtype=np.float32)
    parity_module.assert_exact(array, array.copy())
    parity_module.assert_exact(np.float64(np.nan), np.float64(np.nan))
    parity_module.assert_exact(torch.from_numpy(array), torch.from_numpy(array.copy()))


@pytest.mark.parametrize("change", ["value", "dtype", "shape", "nan", "inf"])
def test_exact_rejects_differences(change: str, parity_module) -> None:
    """Do not hide one-ULP changes, missing values, Inf signs or type changes."""
    left = np.array([1, np.nan, np.inf], dtype=np.float32)
    right = left.copy()
    if change == "value":
        right[0] = np.nextafter(right[0], np.float32(2))
    elif change == "dtype":
        right = right.astype(np.float64)
    elif change == "shape":
        right = right.reshape(1, 3)
    elif change == "nan":
        right[1] = 0
    else:
        right[2] = -np.inf
    with pytest.raises(AssertionError):
        parity_module.assert_exact(left, right)


def test_h5_attributes_are_compared(tmp_path, parity_module) -> None:
    """Recursion checks nested groups and attributes, not only top-level arrays."""
    import h5py

    for name in ("left", "right"):
        with h5py.File(tmp_path / name, "w") as f:
            d = f.create_group("group").create_dataset("values", data=[1, 2])
            d.attrs["label"] = "same"
    with h5py.File(tmp_path / "right", "a") as f:
        f["group/values"].attrs["label"] = "different"
    with h5py.File(tmp_path / "left") as a, h5py.File(tmp_path / "right") as b:
        with pytest.raises(AssertionError, match="attrs"):
            parity_module.compare_h5(
                a, b, {"datasets": 0, "nan_values": 0, "inf_values": 0}
            )
