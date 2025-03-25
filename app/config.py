import os
from dotenv import load_dotenv

load_dotenv()


S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_REGION = os.getenv('S3_REGION')

if not all([S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME, S3_REGION]):
    raise RuntimeError("One or more S3 environment variables are missing.")
