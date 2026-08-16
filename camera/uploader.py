import os
import time
import logging
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import Optional

logger = logging.getLogger("camera.uploader")


class S3VideoUploader:
    """
    Robust chunked uploader for transferring captured match videos from camera SSD to S3/MinIO.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = "http://localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket_name: str = "gaffer-matches",
        region: str = "us-east-1",
        use_ssl: bool = False,
    ):
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Create bucket in MinIO/S3 if it does not already exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Created S3 bucket '{self.bucket_name}'")
            except Exception as e:
                logger.warning(f"Could not auto-create bucket: {e}")

    def upload_file(self, local_path: str, object_key: str, max_retries: int = 5) -> str:
        """
        Upload local MP4 file to S3 with exponential backoff retries.
        Returns the accessible S3/MinIO URL.
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File {local_path} does not exist.")

        file_size = os.path.getsize(local_path)
        logger.info(f"Starting upload of {local_path} ({file_size / (1024 * 1024):.2f} MB) -> s3://{self.bucket_name}/{object_key}")

        for attempt in range(1, max_retries + 1):
            try:
                # Use transfer config for automatic multi-part upload on larger files
                transfer_config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=8 * 1024 * 1024,
                    max_concurrency=4,
                    multipart_chunksize=8 * 1024 * 1024,
                    use_threads=True,
                )

                self.s3_client.upload_file(
                    Filename=local_path,
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Config=transfer_config,
                )

                # Format return URL
                if self.endpoint_url:
                    url = f"{self.endpoint_url.rstrip('/')}/{self.bucket_name}/{object_key}"
                else:
                    url = f"https://{self.bucket_name}.s3.amazonaws.com/{object_key}"

                logger.info(f"Upload successful: {url}")
                return url

            except Exception as e:
                logger.warning(f"Upload attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to upload {local_path} after {max_retries} retries.")
