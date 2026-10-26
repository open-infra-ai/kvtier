#!/usr/bin/env python3
"""Compare host/device addresses for cudaHostRegister vs torch pinned memory.

SGLang's HiCache "kernel" IO backend passes host buffer addresses directly to
CUDA kernels. That is only valid when cudaHostGetDevicePointer(host) == host.
This script checks the two allocation paths on the machine used for the run.
"""

import ctypes

import torch

lib = ctypes.CDLL("libcudart.so")
lib.cudaHostGetDevicePointer.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.c_uint,
]
lib.cudaHostGetDevicePointer.restype = ctypes.c_int


def device_pointer(host_pointer: int) -> tuple[int, int | None]:
    device = ctypes.c_void_p()
    rc = lib.cudaHostGetDevicePointer(
        ctypes.byref(device), ctypes.c_void_p(host_pointer), 0
    )
    return int(rc), device.value


def main() -> None:
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(
        "gpu="
        + torch.cuda.get_device_name(0)
        + f" cc={torch.cuda.get_device_capability(0)}"
    )

    registered = torch.empty(1 << 20, dtype=torch.uint8)
    rc_register = int(
        torch.cuda.cudart().cudaHostRegister(
            registered.data_ptr(), registered.numel(), 0
        )
    )
    rc_get, registered_device = device_pointer(registered.data_ptr())
    print(
        "cudaHostRegister: "
        f"rc_register={rc_register} rc_get_device_pointer={rc_get} "
        f"host={registered.data_ptr():#x} dev={registered_device:#x} "
        f"identity={registered_device == registered.data_ptr()}"
    )
    torch.cuda.cudart().cudaHostUnregister(registered.data_ptr())

    pinned = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
    rc_get, pinned_device = device_pointer(pinned.data_ptr())
    print(
        "torch pin_memory: "
        f"rc_get_device_pointer={rc_get} "
        f"host={pinned.data_ptr():#x} dev={pinned_device:#x} "
        f"identity={pinned_device == pinned.data_ptr()}"
    )


if __name__ == "__main__":
    main()
