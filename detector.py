from ultralytics import YOLO
import cv2

# Load YOLOv8 model (general object detector, works decently for faces)
model = YOLO("yolov8n.pt")

def detect_face(frame):
    results = model.predict(source=frame, conf=0.75, verbose=False)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = frame[y1:y2, x1:x2]
            return face_crop  # Return first face found
    return None
