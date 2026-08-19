import pytest
from app.accel import get_optimal_device, DeviceManager


def test_get_optimal_device_forced_cpu():
    device, dtype = get_optimal_device("cpu")
    assert device == "cpu"
    assert dtype == "float32"


def test_get_optimal_device_forced_cuda():
    device, dtype = get_optimal_device("cuda")
    assert device == "cuda"
    assert dtype == "float16"


def test_device_manager_initialization():
    manager = DeviceManager(requested_device="cpu")
    assert manager.device == "cpu"
    assert manager.dtype == "float32"
