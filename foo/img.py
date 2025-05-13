import base64
import mimetypes

# import binascii # Removed as no longer used
import os
import re
import secrets
import string
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from PIL import Image


# Define the result object
@dataclass
class S3UploadResult:
    success: bool
    url: Optional[str] = None
    s3_uri: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = (
        None  # For non-critical messages like URL construction failure
    )


def image_to_b64(img: Image.Image, image_format: str = "PNG") -> str:
    """Converts a PIL Image to a base64-encoded string with MIME type included.

    Args:
        img (Image.Image): The PIL Image object to convert.
        image_format (str): The format to use when saving the image (e.g., 'PNG', 'JPEG').

    Returns:
        str: A base64-encoded string of the image with MIME type.
    """
    buffer = BytesIO()
    img.save(buffer, format=image_format)
    image_data = buffer.getvalue()
    buffer.close()

    mime_type = f"image/{image_format.lower()}"
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def b64_to_image(base64_str: str) -> Image.Image:
    """Converts a base64 string to a PIL Image object.

    Args:
        base64_str (str): The base64 string, potentially with MIME type as part of a data URI.

    Returns:
        Image.Image: The converted PIL Image object.
    """
    # Strip the MIME type prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image


def upload_pil_image_to_s3(
    pil_image: Image.Image,
    bucket_name: str,
    object_name: str,
    region_name: Optional[str] = None,
    generate_random_filename: bool = False,
) -> S3UploadResult:
    """
    Uploads a PIL.Image object to an S3 bucket, always converting to PNG format.

    The input PIL.Image object is converted to PNG format before uploading.

    Credentials are automatically sourced from standard AWS environment variables
    or the AWS credentials file.

    Args:
        pil_image (Image.Image): The PIL Image object to upload.
        bucket_name (str): The name of the S3 bucket.
        object_name (str): If `generate_random_filename` is True, this is treated as a
                           prefix (e.g., "images/") for the randomly generated PNG filename.
                           If False, this is used as the base for the S3 object key, and its
                           extension will be changed to ".png".
        region_name (str, optional): AWS region name. If not provided,
                                     it attempts to determine it from the bucket or
                                     the default AWS configuration.
        generate_random_filename (bool, optional): If True, generates a random filename
                                     ending in ".png". `object_name` is used as a prefix.
                                     Defaults to False.

    Returns:
        S3UploadResult: An object containing the details of the upload attempt.
    """
    final_mime_type = "image/png"

    try:
        # Convert PIL Image to PNG in-memory
        # The input `pil_image` is already a PIL.Image object
        png_buffer = BytesIO()
        pil_image.save(png_buffer, format="PNG")  # Directly use the input pil_image
        image_data_bytes = png_buffer.getvalue()
        png_buffer.close()

        image_bytes_io = BytesIO(image_data_bytes)

        s3_object_key: str
        if generate_random_filename:
            prefix = object_name
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            random_string = uuid.uuid4().hex
            s3_object_key = f"{prefix}{random_string}.png"
        else:
            base, _ = os.path.splitext(object_name)
            s3_object_key = f"{base}.png"

        s3_client_args = {}
        if region_name:
            s3_client_args["region_name"] = region_name

        s3_client = boto3.client("s3", **s3_client_args)

        s3_client.upload_fileobj(
            Fileobj=image_bytes_io,
            Bucket=bucket_name,
            Key=s3_object_key,
            ExtraArgs={"ContentType": final_mime_type, "ACL": "public-read"},
        )

        s3_uri = f"s3://{bucket_name}/{s3_object_key}"

        current_region_for_url = region_name
        if not current_region_for_url:
            try:
                bucket_location = s3_client.get_bucket_location(Bucket=bucket_name)
                current_region_for_url = (
                    bucket_location.get("LocationConstraint") or "us-east-1"
                )
            except ClientError as e:
                return S3UploadResult(
                    success=True,
                    s3_uri=s3_uri,
                    message=(
                        "Upload successful. Could not determine bucket region "
                        "to construct HTTPS URL. S3 URI is provided. "
                        f"GetBucketLocation error: {str(e)}"
                    ),
                )

        if current_region_for_url == "us-east-1":
            object_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_object_key}"
        else:
            object_url = f"https://{bucket_name}.s3.{current_region_for_url}.amazonaws.com/{s3_object_key}"

        return S3UploadResult(success=True, url=object_url, s3_uri=s3_uri)

    except (NoCredentialsError, PartialCredentialsError):
        return S3UploadResult(
            success=False, error="AWS credentials not found or incomplete."
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message", str(e))
        return S3UploadResult(
            success=False, error=f"S3 Client Error ({error_code}): {error_message}"
        )
    except IOError as e:  # From PIL image.save operation
        return S3UploadResult(
            success=False, error=f"Could not convert image to PNG: {str(e)}"
        )
    except Exception as e:
        return S3UploadResult(
            success=False, error=f"An unexpected error occurred: {str(e)}"
        )


def parse_image_data(image_data_str: str):
    """Parses the image data URL to extract the MIME type and base64 data."""

    data_url_pattern = re.compile(
        r"data:(?P<mime_type>[^;]+);base64,(?P<base64_data>.+)"
    )
    match = data_url_pattern.match(image_data_str)
    if not match:
        raise ValueError("Invalid image data format")
    mime_type = match.group("mime_type")
    base64_data = match.group("base64_data")
    return mime_type, base64_data


def generate_random_suffix(length: int = 24) -> str:
    """Generates a random suffix for the image file name."""
    return "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


def upload_image_to_gcs(pil_image: Image.Image, desired_format: str = "PNG") -> str:
    """Uploads a PIL Image to Google Cloud Storage and returns the public URL.

    Args:
        pil_image (Image.Image): The PIL Image object to upload.
        desired_format (str): The format to save the image in (e.g., 'PNG', 'JPEG').
                              This also determines the MIME type. Defaults to "PNG".
    """
    # Convert PIL Image to bytes and determine mime type
    buffer = BytesIO()
    pil_image.save(buffer, format=desired_format)
    image_data_bytes = buffer.getvalue()
    buffer.close()
    mime_type_str = f"image/{desired_format.lower()}"

    sa_json_content = os.getenv("GCS_SA_JSON")
    sa_temp_file_path = None  # Path for SA JSON if created from string

    if not sa_json_content:
        try:
            storage_client = storage.Client()
        except DefaultCredentialsError:
            raise ValueError(
                "No GCS credentials in the environment and environment variable 'GCS_SA_JSON' not set."
            )
    else:
        # Check if the service account JSON is a path or a JSON string
        if sa_json_content.startswith("{"):
            # Assume it's a JSON string, write to a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".json", encoding="utf-8"
            ) as temp_sa_file:
                temp_sa_file.write(sa_json_content)
                sa_temp_file_path = temp_sa_file.name
            credentials_path = sa_temp_file_path
        else:
            # Assume it's a path to a JSON file
            credentials_path = sa_json_content

        storage_client = storage.Client.from_service_account_json(credentials_path)

    bucket_name = os.getenv("STORAGE_BUCKET")
    if not bucket_name:
        if sa_temp_file_path and os.path.exists(
            sa_temp_file_path
        ):  # Clean up SA temp file if bucket not set
            os.remove(sa_temp_file_path)
        raise ValueError("Environment variable STORAGE_BUCKET not set")

    bucket = storage_client.bucket(bucket_name)

    random_suffix = generate_random_suffix()
    extension = mimetypes.guess_extension(mime_type_str)
    if not extension:  # Fallback if mime type is not recognized by mimetypes
        extension = f".{desired_format.lower()}"

    blob_name = f"images/{random_suffix}{extension}"
    blob = bucket.blob(blob_name)

    image_temp_file_path = None
    public_url_val = ""

    try:
        # Create a temporary file to write the image data
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=extension, mode="wb"
        ) as temp_img_file:
            temp_img_file.write(image_data_bytes)
            image_temp_file_path = temp_img_file.name

        # Upload the temporary file to Google Cloud Storage
        blob.upload_from_filename(image_temp_file_path)
        blob.content_type = mime_type_str
        # blob.make_public()  # Make the blob publicly readable
        # Assuming public_url works as intended (blob is made public by bucket policy or default ACLs)
        public_url_val = blob.public_url

    finally:
        # Clean up image temporary file
        if image_temp_file_path and os.path.exists(image_temp_file_path):
            os.remove(image_temp_file_path)

        # Clean up service account temp file if created
        if sa_temp_file_path and os.path.exists(sa_temp_file_path):
            os.remove(sa_temp_file_path)

    return public_url_val
