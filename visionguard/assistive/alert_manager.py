import sys
import time
import queue
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try importing pyttsx3 safely
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception as e:
    logger.warning(f"pyttsx3 initialization notice: {e}")
    PYTTSX3_AVAILABLE = False

class AlertManager:
    """Non-blocking, thread-safe audio & speech alert manager."""

    def __init__(self, voice_enabled: bool = True, beep_enabled: bool = True, speech_rate: int = 175):
        self.voice_enabled = voice_enabled and PYTTSX3_AVAILABLE
        self.beep_enabled = beep_enabled
        self.speech_rate = speech_rate
        
        self.speech_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        if self.voice_enabled:
            self._start_speech_worker()

    def _start_speech_worker(self):
        """Start daemon thread for non-blocking speech synthesis."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._speech_worker_loop, daemon=True)
        self.worker_thread.start()

    def _speech_worker_loop(self):
        """Background thread loop consuming speech messages from queue."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.speech_rate)
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine in worker thread: {e}")
            return

        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.5)
                if text is None:
                    break
                logger.info(f"[SPEECH ALERTS]: {text}")
                engine.say(text)
                engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech synthesis error: {e}")

    def speak(self, message: str):
        """Queue text message for background non-blocking speech synthesis."""
        if not self.voice_enabled or not message:
            return
        # Avoid queue buildup
        if self.speech_queue.qsize() < 3:
            self.speech_queue.put(message)

    def play_beep(self, threat_type: str = "MEDIUM"):
        """Play non-blocking audio beep based on threat severity."""
        if not self.beep_enabled:
            return

        def _beep_task():
            if sys.platform.startswith('win'):
                try:
                    import winsound
                    if "CRITICAL" in threat_type:
                        winsound.Beep(1500, 300) # High pitch for head level / critical
                    elif "HIGH" in threat_type or "VEHICLE" in threat_type:
                        winsound.Beep(900, 200)  # Mid-high pitch
                    else:
                        winsound.Beep(500, 300)  # Low pitch
                except Exception as e:
                    logger.debug(f"Beep audio error: {e}")
            else:
                # Fallback for Linux / macOS terminal bell
                sys.stdout.write('\a')
                sys.stdout.flush()

        threading.Thread(target=_beep_task, daemon=True).start()

    def stop(self):
        """Stop alert manager threads."""
        self.running = False
        if self.voice_enabled:
            self.speech_queue.put(None)
