"""
VisionGuard Services Subsystem — Weather API Service & Structured Logging
"""

from .weather import WeatherService
from .logger import setup_logger

__all__ = [
    "WeatherService",
    "setup_logger",
]
