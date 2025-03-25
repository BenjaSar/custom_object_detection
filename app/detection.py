from datetime import datetime
import json
from pathlib import Path
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from s3_utils import upload_to_s3
import tempfile
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

CONFIDENCE_THRESHOLD = 0.96

model_path = Path('models') / 'best.pt'

try:
    model = YOLO(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")


def process_image(image):
    """Processes an image and performs object detection."""
    detections = model.predict(image)
    object_detections = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    image_modified = None  # Store modified image for low-confidence cases
    filename_not_detected = "not_detected.jpg"

    for result in detections:
        for box in result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            if confidence < CONFIDENCE_THRESHOLD:
                # Modify image only once
                if image_modified is None:
                    image_modified = result.orig_img.copy()

                # Convert to PIL format for better text rendering
                pil_img = Image.fromarray(cv2.cvtColor(image_modified, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                # Load font (fallback to default if not found)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()

                # Draw bounding box and label
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 10), "unknown", fill="red", font=font)

                # Convert back to OpenCV format
                image_modified = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            else:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                label = model.names.get(cls, "Silobolsa") 

                object_detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

    # If low-confidence detections exist, save & upload the modified image
    if image_modified is not None:
        cv2.imwrite(filename_not_detected, image_modified)

        image_key = f"{timestamp}_not_detected.jpg"
        json_key = f"{timestamp}_not_detections.json"

        image_url = upload_to_s3(filename_not_detected, image_key)

        # Save JSON response
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_json_file:
            json.dump({"Detection result": "No object detected."}, temp_json_file)
            temp_json_path = temp_json_file.name

        json_url = upload_to_s3(temp_json_path, json_key, content_type='application/json')

        # Clean up temporary JSON file
        Path(temp_json_path).unlink()

        return JSONResponse(
            content={"message": "No object detected", "image_url": image_url, "json_url": json_url},
            status_code=200
        )

    # If confident detections exist, save & upload results
    if object_detections:
        output_path = "detected.jpg"
        result.save(filename= output_path)

        image_key = f"{timestamp}_detected.jpg"
        json_key = f"{timestamp}_detections.json"

        image_url = upload_to_s3(output_path, image_key)

        # Save JSON data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_json_file:
            json.dump({"detections": object_detections}, temp_json_file)
            temp_json_path = temp_json_file.name

        json_url = upload_to_s3(temp_json_path, json_key, content_type='application/json')

        # Clean up the temporary JSON file
        Path(temp_json_path).unlink()

        return JSONResponse(
            content={"image_url": image_url, "json_url": json_url, "detections": object_detections}
        )

    return JSONResponse(content={"message": "No confident detections."}, status_code=200)