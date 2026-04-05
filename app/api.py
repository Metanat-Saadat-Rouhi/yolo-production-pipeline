from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI(title="YOLO Production API")
model = YOLO("yolov8n.pt") # Global load for efficiency

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.asarray(image)
    
    # Inference
    results = model.predict(img_array, conf=0.25)
    
    # Format output
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })
            
    return {"count": len(detections), "detections": detections}