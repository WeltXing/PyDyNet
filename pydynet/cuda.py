import numpy as np
import warnings

try:
    import cupy as cp
    cuda_available: bool = True
except ModuleNotFoundError:
    warnings.warn(
        "Cupy is not installed. You can install it with:\n"
        "  pip install cupy-cuda12x  # or appropriate version for your CUDA",
        category=UserWarning)
    cuda_available: bool = False
    cp = object()


def is_available() -> bool:
    return cuda_available


def device_count() -> int:
    if is_available():
        return cp.cuda.runtime.getDeviceCount()
    else:
        return 0


def current_device() -> int:
    return cp.cuda.runtime.getDevice()


def set_device(device: int) -> None:
    return cp.cuda.runtime.setDevice(device)


class Device:

    def __init__(self, device=None) -> None:
        if isinstance(device, str):
            if device == "cpu":
                self.device = "cpu"
            elif device[:4] == "cuda":
                self.device = "cuda"
                if len(device) == 4:
                    device += ':0'

                cuda_id = device.split(':')[-1]
                if not cuda_id.isdigit():
                    raise ValueError(f'Wrong cuda id \"{cuda_id}\"!')

                self.device_id = int(cuda_id)
            else:
                raise ValueError(f"Unknown device \"{device}\"!")

        elif isinstance(device, int):
            self.device = "cuda"
            self.device_id = device

        elif device is None:
            self.device = "cpu"

        elif isinstance(device, Device):
            self.device = device.device
            if self.device != "cpu":
                self.device_id = device.device_id

        if self.device == "cuda":
            if not is_available():
                raise RuntimeError(
                    "Cuda device is not supported on this system.")
            self.device = cp.cuda.Device(self.device_id)
        assert self.device == "cpu" or is_available()

    def __repr__(self) -> str:
        if self.device == "cpu":
            return "Device(type='cpu')"
        else:
            return "Device(type='cuda', index={})".format(self.device_id)

    def __eq__(self, device) -> bool:
        if not isinstance(device, Device):
            device = Device(device)
        if self.device == "cpu":
            return device.device == "cpu"
        else:
            if device.device == "cpu":
                return False
            return self.device == device.device

    @property
    def xp(self):
        return np if self.device == "cpu" else cp

    def __enter__(self):
        if self.device != "cpu" and self.device_id != current_device():
            return self.device.__enter__()

    def __exit__(self, type, value, trace):
        if self.device != "cpu" and self.device_id != current_device():
            return self.device.__exit__(type, value, trace)
