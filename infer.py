import os
import json
from ultralytics import YOLO

# Hardcoded paths
INPUT_FOLDER = "data/demo_images"
OUTPUT_JSON = "outputs/submission_detection_1.json"
MODEL_PATH = "outputs/qr_detector.pt"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Prepare submission list
submission = []

# List all image files
img_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in img_files:
    img_path = os.path.join(INPUT_FOLDER, img_file)
    results = model.predict(img_path, imgsz=640, device=0, conf=0.5)  # adjust conf as needed

    qrs = []
    for r in results:
        boxes = r.boxes.xyxy.tolist()  # xyxy format
        for box in boxes:
            qrs.append({"bbox": [round(b, 2) for b in box]})

    submission.append({
        "image_id": os.path.splitext(img_file)[0],
        "qrs": qrs
    })

# Save JSON
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(submission, f, indent=2)

print(f"\n✅ Inference complete. JSON saved at {OUTPUT_JSON}")
