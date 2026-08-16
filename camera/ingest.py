import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from camera_api import GafferCloudClient
from uploader import S3VideoUploader
from mock_camera import record_mock_match

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [GafferDaemon]: %(message)s",
)
logger = logging.getLogger("gaffer.camera")


class GafferCameraDaemon:
    """
    On-camera service running on embedded Linux OS:
    1. Boots up and pairs with Cloud API.
    2. Sends periodic heartbeat pings.
    3. Listens/polls for match start commands.
    4. Records H.264 stream locally to SSD storage.
    5. On match stop/timeout, uploads MP4 to S3.
    6. Calls Cloud webhook to trigger automated CV match analytics.
    """

    def __init__(self):
        self.serial_number = os.getenv("CAMERA_SERIAL", "GAFFER-CAM-DEMO-001")
        self.turf_name = os.getenv("TURF_NAME", "Main Pitch")
        self.turf_id = os.getenv("TURF_ID", "turf_001")
        self.cloud_api_url = os.getenv("CLOUD_API_URL", "http://localhost:8000/api")
        self.storage_path = os.getenv("STORAGE_PATH", "/tmp/gaffer/camera_storage")
        self.poll_interval_sec = int(os.getenv("POLL_INTERVAL_SEC", "5"))

        os.makedirs(self.storage_path, exist_ok=True)

        self.cloud_client = GafferCloudClient(base_url=self.cloud_api_url)
        self.uploader = S3VideoUploader(
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
            access_key=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            bucket_name=os.getenv("S3_BUCKET_NAME", "gaffer-matches"),
            region=os.getenv("S3_REGION", "us-east-1"),
        )
        self.camera_id: Optional[str] = os.getenv("CAMERA_ID")
        self.api_key: Optional[str] = os.getenv("CAMERA_API_KEY")

    def initialize(self):
        """Register / pair with cloud."""
        logger.info(f"Initializing Gaffer Camera Daemon (Serial: {self.serial_number})...")
        try:
            reg_info = self.cloud_client.register(
                serial_number=self.serial_number,
                turf_name=self.turf_name,
                turf_id=self.turf_id,
            )
            self.camera_id = reg_info["id"]
            self.api_key = reg_info["api_key"]
            logger.info(f"Camera registered successfully! ID={self.camera_id}")
        except Exception as e:
            logger.warning(f"Registration failed ({e}). Will retry during main loop.")

    def run_match_recording(self, match_id: str, duration_sec: int = 5):
        """Record video to local disk, upload to S3, and notify cloud."""
        logger.info(f"===> STARTING RECORDING FOR MATCH {match_id} (Duration: {duration_sec}s) <===")
        local_file = os.path.join(self.storage_path, f"{match_id}.mp4")

        # 1. Capture video locally
        record_mock_match(local_file, duration_sec=duration_sec, fps=30)
        file_size = os.path.getsize(local_file)

        # 2. Upload to S3
        object_key = f"matches/{match_id}/{match_id}.mp4"
        logger.info(f"Uploading {local_file} to cloud storage...")
        s3_url = self.uploader.upload_file(local_file, object_key)

        # 3. Notify cloud webhook
        logger.info("Notifying Gaffer Cloud API of upload completion...")
        response = self.cloud_client.notify_upload_complete(
            camera_id=self.camera_id,
            match_id=match_id,
            s3_url=s3_url,
            duration_sec=duration_sec,
            file_size_bytes=file_size,
        )
        logger.info(f"Cloud response: {response}")
        logger.info(f"===> MATCH {match_id} COMPLETED & SUBMITTED FOR AI ANALYSIS <===")

    def start(self):
        """Main daemon loop."""
        self.initialize()
        last_heartbeat_time = 0

        logger.info("Gaffer Camera Daemon is running and listening for match triggers...")
        while True:
            try:
                # 1. Re-register if needed
                if not self.camera_id:
                    self.initialize()
                    time.sleep(self.poll_interval_sec)
                    continue

                # 2. Send periodic heartbeat
                now = time.time()
                if now - last_heartbeat_time >= 30:
                    try:
                        self.cloud_client.send_heartbeat(self.camera_id, status="ONLINE")
                        last_heartbeat_time = now
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")

                # 3. Poll for active match assigned to this camera
                poll_res = self.cloud_client.check_next_match(self.camera_id)
                if poll_res.get("has_active_match"):
                    match_data = poll_res["match"]
                    match_id = match_data["id"]
                    self.run_match_recording(match_id, duration_sec=5)

            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")

            time.sleep(self.poll_interval_sec)


if __name__ == "__main__":
    daemon = GafferCameraDaemon()
    daemon.start()
