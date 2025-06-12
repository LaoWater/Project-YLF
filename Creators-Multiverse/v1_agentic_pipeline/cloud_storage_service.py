# cloud_storage_service.py
import os
import json
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
import mimetypes
from dotenv import load_dotenv
load_dotenv()  # if you're using .env

# Configuration - these should be in your config.py or environment variables
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-social-media-bucket")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "your-project-id")

# Path to service account key file (optional if using default credentials)
GCS_SERVICE_ACCOUNT_KEY_PATH = os.getenv("GCS_SERVICE_ACCOUNT_KEY_PATH")


# Load the .env file from the current directory
load_dotenv()

print("--- Running Environment & Path Check ---")

# 1. Check if the environment variable is loaded
key_path = os.getenv("GCS_SERVICE_ACCOUNT_KEY_PATH")
if key_path:
    print(f"✅ Environment variable 'GCS_SERVICE_ACCOUNT_KEY_PATH' is loaded.")
    print(f"   Value: '{key_path}'")
else:
    print(f"❌ ERROR: Environment variable 'GCS_SERVICE_ACCOUNT_KEY_PATH' is NOT loaded. Check your .env file and its location.")
    exit()

# 2. Check if the path exists from the script's perspective
print("\n--- Checking Path ---")
path_exists = os.path.exists(key_path)
if path_exists:
    print(f"✅ The path '{key_path}' EXISTS.")
else:
    print(f"❌ CRITICAL ERROR: The path '{key_path}' DOES NOT EXIST from the script's perspective.")
    print("   This is the reason your authentication is failing.")
    print("   Common causes: Docker volume not mounted, incorrect path, or permissions issue.")

# 3. Check for read permissions (if the path exists)
if path_exists:
    print("\n--- Checking Read Permissions ---")
    try:
        with open(key_path, 'r') as f:
            f.read(10) # Try to read a few bytes
        print(f"✅ The script has permission to READ the file at '{key_path}'.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: The path exists, but the script CANNOT READ the file.")
        print(f"   Error details: {e}")

print("\n--- End of Check ---")


class CloudStorageService:
    """Service for uploading and managing files in Google Cloud Storage."""

    def __init__(self, bucket_name: str = GCS_BUCKET_NAME, project_id: str = GCS_PROJECT_ID):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._client = None
        self._bucket = None

    def _get_client(self) -> storage.Client:
        """Get or create the GCS client."""
        if self._client is None:
            try:
                if GCS_SERVICE_ACCOUNT_KEY_PATH and os.path.exists(GCS_SERVICE_ACCOUNT_KEY_PATH):
                    self._client = storage.Client.from_service_account_json(
                        GCS_SERVICE_ACCOUNT_KEY_PATH,
                        project=self.project_id
                    )
                else:
                    # Use default credentials (works in GCP environments or with gcloud auth)
                    self._client = storage.Client(project=self.project_id)

            except Exception as e:
                raise Exception(f"Failed to initialize GCS client: {e}")

        return self._client

    def _get_bucket(self) -> storage.Bucket:
        """Get or create the GCS bucket reference."""
        if self._bucket is None:
            client = self._get_client()
            self._bucket = client.bucket(self.bucket_name)
        return self._bucket

    async def upload_file(
            self,
            local_file_path: str,
            cloud_file_path: str,
            content_type: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to Google Cloud Storage.

        Args:
            local_file_path: Path to the local file to upload
            cloud_file_path: Destination path in the cloud bucket
            content_type: MIME type of the file (auto-detected if not provided)
            metadata: Additional metadata to attach to the file

        Returns:
            Dictionary containing upload result information
        """
        try:
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Local file not found: {local_file_path}")

            bucket = self._get_bucket()
            blob = bucket.blob(cloud_file_path)

            # Auto-detect content type if not provided
            if content_type is None:
                content_type, _ = mimetypes.guess_type(local_file_path)
                if content_type is None:
                    content_type = 'application/octet-stream'

            # Set metadata
            if metadata:
                blob.metadata = metadata

            # Upload the file
            with open(local_file_path, 'rb') as file_data:
                blob.upload_from_file(file_data, content_type=content_type)

            # Get public URL (assumes bucket allows public access)
            public_url = f"https://storage.googleapis.com/{self.bucket_name}/{cloud_file_path}"

            upload_result = {
                "success": True,
                "cloud_path": cloud_file_path,
                "public_url": public_url,
                "bucket_name": self.bucket_name,
                "content_type": content_type,
                "size": os.path.getsize(local_file_path),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "local_path": local_file_path
            }

            print(f"✅ Successfully uploaded {local_file_path} to {cloud_file_path}")
            return upload_result

        except GoogleCloudError as e:
            error_msg = f"GCS upload failed for {local_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "local_path": local_file_path,
                "cloud_path": cloud_file_path
            }
        except Exception as e:
            error_msg = f"Upload failed for {local_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "local_path": local_file_path,
                "cloud_path": cloud_file_path
            }

    async def upload_text_content(
            self,
            text_content: str,
            cloud_file_path: str,
            metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload text content directly to Google Cloud Storage without saving locally first.

        Args:
            text_content: The text content to upload
            cloud_file_path: Destination path in the cloud bucket
            metadata: Additional metadata to attach to the file

        Returns:
            Dictionary containing upload result information
        """
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(cloud_file_path)

            # Set metadata
            if metadata:
                blob.metadata = metadata

            # Upload text content directly
            blob.upload_from_string(text_content, content_type='text/plain; charset=utf-8')

            # Get public URL
            public_url = f"https://storage.googleapis.com/{self.bucket_name}/{cloud_file_path}"

            upload_result = {
                "success": True,
                "cloud_path": cloud_file_path,
                "public_url": public_url,
                "bucket_name": self.bucket_name,
                "content_type": "text/plain; charset=utf-8",
                "size": len(text_content.encode('utf-8')),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "content_preview": text_content[:100] + "..." if len(text_content) > 100 else text_content
            }

            print(f"✅ Successfully uploaded text content to {cloud_file_path}")
            return upload_result

        except GoogleCloudError as e:
            error_msg = f"GCS text upload failed for {cloud_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "cloud_path": cloud_file_path
            }
        except Exception as e:
            error_msg = f"Text upload failed for {cloud_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "cloud_path": cloud_file_path
            }

    async def upload_json_data(
            self,
            json_data: Dict[Any, Any],
            cloud_file_path: str,
            metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload JSON data directly to Google Cloud Storage.

        Args:
            json_data: The data to serialize and upload as JSON
            cloud_file_path: Destination path in the cloud bucket
            metadata: Additional metadata to attach to the file

        Returns:
            Dictionary containing upload result information
        """
        try:
            json_content = json.dumps(json_data, indent=2, ensure_ascii=False)

            bucket = self._get_bucket()
            blob = bucket.blob(cloud_file_path)

            # Set metadata
            if metadata:
                blob.metadata = metadata

            # Upload JSON content
            blob.upload_from_string(json_content, content_type='application/json; charset=utf-8')

            # Get public URL
            public_url = f"https://storage.googleapis.com/{self.bucket_name}/{cloud_file_path}"

            upload_result = {
                "success": True,
                "cloud_path": cloud_file_path,
                "public_url": public_url,
                "bucket_name": self.bucket_name,
                "content_type": "application/json; charset=utf-8",
                "size": len(json_content.encode('utf-8')),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(json_data) if isinstance(json_data, (list, dict)) else 1
            }

            print(f"✅ Successfully uploaded JSON data to {cloud_file_path}")
            return upload_result

        except GoogleCloudError as e:
            error_msg = f"GCS JSON upload failed for {cloud_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "cloud_path": cloud_file_path
            }
        except Exception as e:
            error_msg = f"JSON upload failed for {cloud_file_path}: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "cloud_path": cloud_file_path
            }

    def generate_cloud_path(self, filename_base: str, platform: str, file_type: str, extension: str) -> str:
        """
        Generate a standardized cloud storage path for files.

        Args:
            filename_base: Base filename (e.g., "hello_world_intro")
            platform: Platform name (e.g., "facebook", "instagram")
            file_type: Type of file (e.g., "text", "image", "summary")
            extension: File extension (e.g., "txt", "png", "json")

        Returns:
            Standardized cloud storage path
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"social_media_posts/{timestamp}/{platform}/{file_type}/{filename_base}.{extension}"


# Initialize a global instance for easy importing
cloud_storage = CloudStorageService()


# --- Utility Functions ---
async def upload_generated_post_files(
        filename_base: str,
        platform: str,
        text_content: str,
        media_file_path: Optional[str] = None,
        media_generation_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload all files related to a generated social media post.

    Returns:
        Dictionary containing upload results for all files
    """
    upload_results = {
        "platform": platform,
        "filename_base": filename_base,
        "uploads": []
    }

    # Upload text content
    text_cloud_path = cloud_storage.generate_cloud_path(filename_base, platform, "text", "txt")
    text_metadata = {
        "platform": platform,
        "content_type": "social_media_text",
        "filename_base": filename_base
    }

    text_result = await cloud_storage.upload_text_content(
        text_content,
        text_cloud_path,
        text_metadata
    )
    upload_results["uploads"].append(text_result)

    # Upload media file if provided
    if media_file_path and os.path.exists(media_file_path):
        _, ext = os.path.splitext(media_file_path)
        ext = ext.lstrip('.')  # Remove the dot

        media_cloud_path = cloud_storage.generate_cloud_path(filename_base, platform, "image", ext)
        media_metadata = {
            "platform": platform,
            "content_type": "social_media_image",
            "filename_base": filename_base,
            "generation_prompt": media_generation_prompt[:500] if media_generation_prompt else ""
        }

        media_result = await cloud_storage.upload_file(
            media_file_path,
            media_cloud_path,
            metadata=media_metadata
        )
        upload_results["uploads"].append(media_result)

    return upload_results