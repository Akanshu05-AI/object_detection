import time
import requests
import threading
import logging
from typing import Optional, Dict, Any
from visionguard.config import settings

logger = logging.getLogger(__name__)

class WeatherService:
    """Async cached weather service for OpenWeatherMap API."""

    def __init__(
        self,
        api_key: str = settings.OPENWEATHER_API_KEY,
        city: str = settings.DEFAULT_CITY,
        refresh_interval_sec: int = settings.WEATHER_REFRESH_SEC
    ):
        self.api_key = api_key
        self.city = city
        self.refresh_interval_sec = refresh_interval_sec

        self._cached_temp: Optional[float] = None
        self._cached_condition: str = "Unknown"
        self._last_fetch_time: float = 0.0
        self._is_fetching: bool = False

    def get_temperature(self) -> Optional[float]:
        """Return cached temperature in Celsius, triggering background update if expired."""
        curr_time = time.time()
        if self._cached_temp is None or (curr_time - self._last_fetch_time > self.refresh_interval_sec):
            self._trigger_background_update()
        return self._cached_temp

    def get_weather_summary(self) -> Dict[str, Any]:
        """Return dict with city, temperature, and condition."""
        temp = self.get_temperature()
        return {
            "city": self.city,
            "temp_celsius": temp if temp is not None else "N/A",
            "condition": self._cached_condition if temp is not None else "Offline/Unavailable",
            "last_updated": round(time.time() - self._last_fetch_time, 1) if self._last_fetch_time > 0 else None
        }

    def _trigger_background_update(self):
        """Spawn daemon thread to fetch weather asynchronously."""
        if self._is_fetching or not self.api_key:
            return
        self._is_fetching = True
        threading.Thread(target=self._fetch_weather_sync, daemon=True).start()

    def _fetch_weather_sync(self):
        """Synchronous HTTP fetch executed in background thread."""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city}&appid={self.api_key}&units=metric"
            resp = requests.get(url, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                if "main" in data and "temp" in data["main"]:
                    self._cached_temp = round(float(data["main"]["temp"]), 1)
                    if "weather" in data and len(data["weather"]) > 0:
                        self._cached_condition = data["weather"][0].get("main", "Clear")
                    self._last_fetch_time = time.time()
                    logger.info(f"Updated weather for {self.city}: {self._cached_temp}°C ({self._cached_condition})")
            else:
                logger.warning(f"Weather API returned status code {resp.status_code}")
        except Exception as e:
            logger.warning(f"Weather API request failed: {e}")
        finally:
            self._is_fetching = False
