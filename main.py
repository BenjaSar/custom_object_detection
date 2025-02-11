## Author : FS
## Date: 2025
import os
import cv2
import numpy as np
import boto3
import json
import logging
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


model_path = Path('models') / 'best.pt'

S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_REGION = os.getenv('S3_REGION')

if not all([S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME, S3_REGION]):
    raise RuntimeError("One or more S3 environment variables are missing.")

app = FastAPI(
    author="FS",
    title="Custom Object Detection API",
    description="Detects silobags in images",
    version="MVP"
)

allowed_origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = 8000

try:
    model = YOLO('best.pt')
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)

CONFIDENCE_THRESHOLD = 0.96

@app.get("/")
async def read_root():
    return {"message": f"API is Running on the port {PORT}"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(..., description="Upload an image file")):
    try:
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        detections = model.predict(image)
        object_detections = []

        for result in detections:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                label = model.names.get(cls, "Silobolsa")

                object_detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        if not object_detections:
            return JSONResponse(content={"message": "No silobags detected."}, status_code=200)

        output_image_path = "detected_silobag.jpg"
        result.save(filename=output_image_path)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_key = f"{timestamp}_detected_image.jpg"
        json_key = f"{timestamp}_detections.json"

        s3_client.upload_file(output_image_path, S3_BUCKET_NAME, image_key)

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=json_key,
            Body=json.dumps({"detections": object_detections}),
            ContentType='application/json'
        )

        image_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{image_key}"
        json_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{json_key}"

        return JSONResponse(content={
            "image_url": image_url,
            "json_url": json_url,
            "detections": object_detections
        })

    except ClientError as e:
        logging.error(f"Error uploading to S3: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)



