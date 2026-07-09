from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load pretrained YOLOv8 nano model (smallest, fastest — good for demo)
# It downloads automatically on first run (~6MB)
model = YOLO('yolov8n.pt')

def detect_objects(image):
    """
    Takes a PIL Image, returns:
    - annotated image with bounding boxes
    - list of detected objects with confidence scores
    """
    # Run detection
    results = model(image)
    
    # Get annotated image (with bounding boxes drawn)
    annotated = results[0].plot()  # returns numpy array
    annotated_pil = Image.fromarray(annotated)
    
    # Extract detection details
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            'object': class_name,
            'confidence': round(confidence * 100, 1),
            'location': f"({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})"
        })
    
    return annotated_pil, detections