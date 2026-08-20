import torch
from typing import Dict, Any

def get_device() -> str:
    """Determine the optimal computing device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_device_info() -> Dict[str, Any]:
    """Retrieve detailed system hardware execution environment info."""
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    gpu_count = 0
    memory_allocated_mb = 0.0
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        try:
            memory_allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        except Exception:
            memory_allocated_mb = 0.0

    return {
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "device_name": device_name,
        "gpu_count": gpu_count,
        "memory_allocated_mb": memory_allocated_mb,
        "pytorch_version": torch.__version__,
    }
