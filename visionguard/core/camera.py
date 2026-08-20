import cv2
import time
import logging
from typing import Union, Tuple, Optional, Generator, Any

logger = logging.getLogger(__name__)

class CameraSource:
    """Unified camera feed abstraction for Webcams, IP Webcams, Video Files & RTSP streams."""
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3
    ):
        self.source = source
        self.target_width = width
        self.target_height = height
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected: bool = False
        self.total_frames_read: int = 0
        self.is_file_source: bool = isinstance(source, str) and not source.startswith(("http://", "https://", "rtsp://"))
        
        self._connect()

    def _connect(self) -> bool:
        """Initialize OpenCV VideoCapture connection."""
        if self.cap is not None:
            self.cap.release()
            
        logger.info(f"Connecting to video source: {self.source}...")
        self.cap = cv2.VideoCapture(self.source)
        
        if self.cap and self.cap.isOpened():
            # Set resolution if supported by source
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.is_connected = True
            logger.info(f"Successfully connected to source {self.source}")
            return True
        else:
            self.is_connected = False
            logger.warning(f"Failed to open video source: {self.source}")
            return False

    def read(self) -> Tuple[bool, Optional[Any]]:
        """Read a single video frame with automatic reconnect handling."""
        if not self.is_connected or self.cap is None:
            if self.auto_reconnect and not self.is_file_source:
                if not self._reconnect():
                    return False, None
            else:
                return False, None

        success, frame = self.cap.read()
        
        if not success:
            if self.auto_reconnect and not self.is_file_source:
                logger.warning("Frame read failed, attempting reconnect...")
                if self._reconnect():
                    success, frame = self.cap.read()
                    
        if success and frame is not None:
            self.total_frames_read += 1
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            return True, frame
            
        return False, None

    def _reconnect(self) -> bool:
        """Attempt reconnection to streaming camera source."""
        for attempt in range(1, self.max_reconnect_attempts + 1):
            logger.info(f"Reconnect attempt {attempt}/{self.max_reconnect_attempts}...")
            time.sleep(1.0)
            if self._connect():
                return True
        self.is_connected = False
        return False

    def get_fps(self) -> float:
        """Get source FPS if available."""
        if self.cap and self.is_connected:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    def get_total_frames(self) -> int:
        """Get total frames for file sources."""
        if self.cap and self.is_connected and self.is_file_source:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def release(self):
        """Cleanly release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        logger.info(f"Released video source {self.source}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
