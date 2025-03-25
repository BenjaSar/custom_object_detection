from datetime import datetime
import json
from pathlib import Path
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from s3_utils import upload_to_s3
import tempfile

CONFIDENCE_THRESHOLD = 0.96

model_path = Path('models') / 'best.pt'
print(model_path)

try:
    model = YOLO(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")


def process_image(image):
    "Function used for processing the image to pass to the model"
    detections = model.predict(image)
    object_detections = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for result in detections:
        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                filename_not_detected = "not_detected.jpg"
                result.save(filename=filename_not_detected)

                image_key = f"{timestamp}_not_detected.jpg"
                json_key = f"{timestamp}_not_detections.json"

                image_url = upload_to_s3(filename_not_detected, image_key)
                #json_data = json.dumps({"Detection result": "No object detected."})
                
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_json_file:
                    json.dump({"Detection result": "No object detected."}, temp_json_file)
                    temp_json_path = temp_json_file.name

                json_url = upload_to_s3(temp_json_path, json_key, content_type='application/json')

                # Clean up the temporary file
                Path(temp_json_path).unlink()

                return JSONResponse(
                    content={"message": "No object detected", "image_url": image_url, "json_url": json_url},
                    status_code=200
                )

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            #label = model.names[cls] if cls < len(model.names) else "Unknown"
            label = model.names.get(cls, "Silobolsa")

            object_detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    if not object_detections:
        return JSONResponse(content={"message": "No confident detections."}, status_code=200)

    output_path = "detected.jpg" 
    result.save(filename=output_path)

    image_key = f"{timestamp}_detected.jpg"
    json_key = f"{timestamp}_detections.json"

    image_url = upload_to_s3(output_path, image_key)
    #json_data = json.dumps({"detections": object_detections})
    #json_url = upload_to_s3(json_data, json_key, content_type='application/json')

     # Save JSON data to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_json_file:
        json.dump({"detections": object_detections}, temp_json_file)
        temp_json_path = temp_json_file.name

    json_url = upload_to_s3(temp_json_path, json_key, content_type='application/json')

    # Clean up the temporary file
    Path(temp_json_path).unlink()

    return JSONResponse(
        content={"image_url": image_url, "json_url": json_url, "detections": object_detections}
    )
