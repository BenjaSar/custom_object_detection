## Author : FS
## Date: January 2025

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import cv2
import numpy as np
from detection import process_image

# Load environment variables
load_dotenv()

# Constants
HOST = os.getenv("HOST", "0.0.0.0")  # Default to 0.0.0.0 if not set
PORT = int(os.getenv("PORT", 8000))  # Convert PORT to int, default to 8000

# FastAPI app instance
app = FastAPI(
    title="Custom Object Detection API",
    description="Detects silobags in images",
    version="MVP",
)

allowed_origins = [
    "http://localhost",
    "http://localhost:8000",
]

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; adjust for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": f"API is Running on port {PORT}"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(..., description="Upload an image file")):
    image_bytes = await file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    return process_image(image)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
