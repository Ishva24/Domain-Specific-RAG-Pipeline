"""
Hardware Acceleration & Device Management
=========================================
Detects and configures hardware acceleration for transformer models
(CUDA, Apple Silicon MPS, or multi-core CPU) with automatic precision selection.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


def get_optimal_device(force_device: Optional[str] = None) -> Tuple[str, str]:
    """Detect and return optimal (device_name, dtype_name).

    Returns:
      (device, dtype) - e.g. ("cuda", "float16"), ("mps", "float32"), or ("cpu", "float32")
    """
    if force_device:
        device = force_device.lower()
        dtype = "float16" if device == "cuda" else "float32"
        return device, dtype

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            # Use bfloat16 on Ampere or newer GPUs, else float16
            if torch.cuda.is_bf16_supported():
                dtype = "bfloat16"
            else:
                dtype = "float16"
            logger.info("Detected CUDA acceleration", device=device, dtype=dtype, device_name=torch.cuda.get_device_name(0))
            return device, dtype

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = "float32"
            logger.info("Detected Apple Silicon MPS acceleration", device=device, dtype=dtype)
            return device, dtype

    except ImportError:
        logger.warning("PyTorch not installed. Defaulting to CPU.")

    logger.info("Using CPU device", device="cpu", dtype="float32")
    return "cpu", "float32"


class DeviceManager:
    """Manages runtime compute device settings and memory caching."""

    def __init__(self, requested_device: Optional[str] = None):
        self.device, self.dtype = get_optimal_device(requested_device)

    def optimize_model(self, model: any) -> any:
        """Move model to optimal device with corresponding precision if supported."""
        try:
            import torch

            if hasattr(model, "to"):
                if self.device == "cuda":
                    dtype_obj = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
                    model = model.to(self.device, dtype=dtype_obj)
                else:
                    model = model.to(self.device)
            return model
        except Exception as e:
            logger.error("Failed to optimize model device placement", error=str(e))
            return model
