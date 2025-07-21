# from ultralytics import YOLO

# # Load a pretrained YOLOv8 model
# model = YOLO('yolov8n.pt')  # 'n' is for nano version (fastest)

# # Run prediction on any sample image
# results = model('/Users/raman/Downloads/WhatsApp Image 2025-07-22 at 01.40.59.jpeg', show=True)

# for r in results:
#     for box in r.boxes:
#         cls_id = int(box.cls[0])
#         label = model.names[cls_id]
#         if label == 'cow':
#             print(f"Cow detected with confidence {box.conf[0]:.2f}")

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pretrained YOLOv8 model (trained on COCO)
model = YOLO('yolov8n.pt')  # You can also try yolov8m.pt or yolov8l.pt later

# Load image
image_path = '/Users/raman/Downloads/WhatsApp Image 2025-07-22 at 01.40.59.jpeg'  # Replace with your actual image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run prediction
results = model(image_path)

def estimate_weight(bbox_width, bbox_height):
    area = bbox_width * bbox_height
    weight = 0.0015 * (4*area)/6  # You can adjust this coefficient
    return round(weight, 2)

# Draw detections with weight
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if label == 'cow':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            weight = estimate_weight(width, height)

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Label with confidence and weight
            cv2.putText(image_rgb, f'{label} {conf:.2f} | ~{weight}kg', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Show result
plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
plt.title("Cow Detection + Estimated Weight")
plt.axis('off')
plt.show()

