from unittest.mock import patch

from fart.model.device import get_device


@patch("torch.backends.mps.is_available", return_value=True)
def test_get_device_returns_mps_when_available(_mock_mps_available: object) -> None:
    device = get_device()

    assert device.type == "mps"


@patch("torch.backends.mps.is_available", return_value=False)
def test_get_device_returns_cpu_when_mps_unavailable(
    _mock_mps_available: object,
) -> None:
    device = get_device()

    assert device.type == "cpu"
