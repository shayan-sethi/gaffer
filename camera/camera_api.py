import os
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger("camera.api")


class GafferCloudClient:
    """
    HTTP client for on-camera daemon to communicate with Gaffer Cloud API.
    """

    def __init__(self, base_url: str = "http://localhost:8000/api", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-Camera-Key": self.api_key})

    def register(self, serial_number: str, turf_name: str, turf_id: Optional[str] = None, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Pair/register camera with cloud backend."""
        url = f"{self.base_url}/cameras/register"
        payload = {
            "serial_number": serial_number,
            "turf_name": turf_name,
            "turf_id": turf_id,
            "ip_address": ip_address,
            "firmware_version": "1.0.0",
        }
        res = self.session.post(url, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        self.api_key = data.get("api_key")
        if self.api_key:
            self.session.headers.update({"X-Camera-Key": self.api_key})
        return data

    def send_heartbeat(self, camera_id: str, status: str = "ONLINE", ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Send periodic keepalive heartbeat."""
        url = f"{self.base_url}/cameras/{camera_id}/heartbeat"
        payload = {"status": status, "ip_address": ip_address}
        res = self.session.post(url, json=payload, timeout=5)
        res.raise_for_status()
        return res.json()

    def check_next_match(self, camera_id: str) -> Dict[str, Any]:
        """Poll for active/pending matches assigned to this camera."""
        url = f"{self.base_url}/cameras/{camera_id}/next_match"
        res = self.session.get(url, timeout=5)
        res.raise_for_status()
        return res.json()

    def notify_upload_complete(self, camera_id: str, match_id: str, s3_url: str, duration_sec: int, file_size_bytes: Optional[int] = None) -> Dict[str, Any]:
        """Signal cloud API that video upload is finished and processing should begin."""
        url = f"{self.base_url}/cameras/{camera_id}/upload_complete"
        payload = {
            "match_id": match_id,
            "s3_url": s3_url,
            "duration_sec": duration_sec,
            "file_size_bytes": file_size_bytes,
        }
        res = self.session.post(url, json=payload, timeout=15)
        res.raise_for_status()
        return res.json()
