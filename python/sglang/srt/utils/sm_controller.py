"""SM Controller for partitioning GPU SMs between draft and target models.

Adapted from BulletServe's sm_controller.py.
Requires libsmctrl.so to be built from BulletServe/csrc.
"""

import ctypes
import ctypes.util
import logging
import os
from functools import wraps

import torch

logger = logging.getLogger(__name__)


def get_num_tpcs() -> int:
    """Get the number of TPCs based on the GPU type."""
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_name = props.name.lower()
        num_sms = props.multi_processor_count

        # Modern GPUs have 2 SMs per TPC
        major = props.major
        if major >= 7:  # Volta and newer
            return num_sms // 2
        else:
            return num_sms
    except Exception:
        return 54  # Default A100 TPCs


def _check_ret_code(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        rets = func(*args, **kwargs)
        if isinstance(rets, tuple):
            ret_code = rets[0]
            ret_res = rets[1]
        else:
            ret_code = rets or 0
            ret_res = None
        if ret_code != 0:
            raise OSError(ret_code, f"{os.strerror(ret_code)} in {func.__name__}")
        return ret_res
    return wrapper


class c_uint128(ctypes.Structure):
    """128-bit unsigned integer for GPUs with >64 TPCs (e.g., H100)."""
    _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_uint64)]

    def __init__(self, value=0):
        super().__init__()
        if isinstance(value, int):
            self.low = value & 0xFFFFFFFFFFFFFFFF
            self.high = (value >> 64) & 0xFFFFFFFFFFFFFFFF
        else:
            self.low = 0
            self.high = 0

    @property
    def value(self):
        return self.low | (self.high << 64)


class _LibSMCtrl:
    """Low-level wrapper for libsmctrl C library."""

    def __init__(self, libsmctrl_path: str):
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.total_sms = device_props.multi_processor_count

        # Try to find the library
        if not os.path.exists(libsmctrl_path):
            # Try common locations (relative to this file: python/sglang/srt/utils/)
            base_dir = os.path.join(os.path.dirname(__file__), "../../../../../")
            search_paths = [
                libsmctrl_path,
                os.path.join(base_dir, "cospec_scripts/libsmctrl/build/libsmctrl.so"),
                "/usr/local/lib/libsmctrl.so",
            ]
            for path in search_paths:
                if os.path.exists(path):
                    libsmctrl_path = path
                    break

        try:
            self.lib = ctypes.CDLL(libsmctrl_path)
        except Exception as e:
            raise OSError(
                f"Failed to load libsmctrl.so from {libsmctrl_path}: {e}\n"
                "Please build it: cd cospec_scripts && bash build_libsmctrl.sh"
            )

    @_check_ret_code
    def set_stream_mask(self, stream: torch.cuda.Stream, mask: int) -> None:
        if self.total_sms < 128:
            return self.lib.libsmctrl_set_stream_mask(
                ctypes.c_void_p(stream.cuda_stream), ctypes.c_uint64(mask)
            )
        else:
            return self.lib.libsmctrl_set_stream_mask_ext(
                ctypes.c_void_p(stream.cuda_stream), c_uint128(mask)
            )

    @_check_ret_code
    def get_tpc_count(self, cuda_dev: int) -> int:
        num_tpcs = ctypes.c_uint32()
        ret = self.lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(num_tpcs), cuda_dev)
        return ret, num_tpcs.value

    @_check_ret_code
    def make_mask(self, low: int, high_exclusive: int) -> int:
        result = ctypes.c_uint64()
        ret = self.lib.libsmctrl_make_mask(ctypes.byref(result), low, high_exclusive)
        return ret, result.value


class SMController:
    """Controller for SM partitioning between draft and target models.

    Usage:
        sm_ctrl = SMController()

        # Partition: draft gets TPCs 0-13, target gets 13-66
        draft_tpcs = int(sm_ctrl.total_tpcs * 0.2)
        sm_ctrl.set_stream_mask(draft_stream, 0, draft_tpcs)
        sm_ctrl.set_stream_mask(target_stream, draft_tpcs, sm_ctrl.total_tpcs)
    """

    def __init__(self, libsmctrl_path: str = "libsmctrl.so", enabled: bool = True):
        self.enabled = enabled

        if not enabled:
            self.total_tpcs = get_num_tpcs()
            logger.info(f"SMController disabled, total_tpcs={self.total_tpcs}")
            return

        try:
            self.lib = _LibSMCtrl(libsmctrl_path)
            self.total_tpcs = self.lib.get_tpc_count(torch.cuda.current_device())
            logger.info(f"SMController initialized, total_tpcs={self.total_tpcs}")
        except Exception as e:
            logger.warning(f"Failed to initialize SMController: {e}. Disabling SM partitioning.")
            self.enabled = False
            self.total_tpcs = get_num_tpcs()

    def set_stream_mask(
        self,
        stream: torch.cuda.Stream,
        low: int,
        high_exclusive: int,
        reversed: bool = False
    ) -> None:
        """Set SM mask for a CUDA stream.

        Args:
            stream: CUDA stream to set mask for
            low: Starting TPC index (inclusive)
            high_exclusive: Ending TPC index (exclusive)
            reversed: If True, use TPCs from the high end instead
        """
        if not self.enabled:
            return

        if reversed:
            low, high_exclusive = self.total_tpcs - high_exclusive, self.total_tpcs - low

        logger.debug(f"SMController: set TPCs {low}-{high_exclusive} ({(high_exclusive-low)*2} SMs) for stream")
        mask = self.lib.make_mask(low, high_exclusive)
        self.lib.set_stream_mask(stream, mask)
