## Author : FS
## Date: January 2025
import os
import boto3
import logging
from fastapi.responses import JSONResponse
from io import BytesIO
from botocore.exceptions import ClientError
from config import S3_BUCKET_NAME, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY
import json
import numpy as np
import cv2


s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)

def upload_to_s3(file_path, key, content_type=None):
    "Function used to upload the images to s3 bucket"
    try:
        if content_type:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=key,
                Body=open(file_path, "rb"),
                ContentType=content_type
            )
        else:
            s3_client.upload_file(file_path, S3_BUCKET_NAME, key)

        return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{key}"
    #except Exception as e:
    #    logging.error(f"Error uploading to S3: {e}")
    #    raise HTTPException(status_code=500, detail="Failed to upload to S3")


    except ClientError as e:
        logging.error(f"Error uploading to S3: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
