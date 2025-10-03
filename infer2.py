import os
import json
import cv2
from ultralytics import YOLO

# Hardcoded paths
INPUT_FOLDER = "data/demo_images"
OUTPUT_JSON = "outputs/submission_decoding_2.json"
MODEL_PATH = "outputs/qr_detector.pt"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Prepare submission list
submission = []

# List all image files
img_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in img_files:
    img_path = os.path.join(INPUT_FOLDER, img_file)
    
    # Run detection
    results = model.predict(img_path, imgsz=640, device=0, conf=0.5)

    # Read image for QR decoding
    img = cv2.imread(img_path)
    qr_detector = cv2.QRCodeDetector()

    qrs = []
    for r in results:
        boxes = r.boxes.xyxy.tolist()  # xyxy format
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            qr_crop = img[y1:y2, x1:x2]

            # Decode QR code
            value, points, _ = qr_detector.detectAndDecode(qr_crop)
            value = value if value else ""  # empty string if decoding fails

            qrs.append({
                "bbox": [x1, y1, x2, y2],
                "value": value
            })

    submission.append({
        "image_id": os.path.splitext(img_file)[0],
        "qrs": qrs
    })

# Save JSON
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(submission, f, indent=2)

print(f"\n✅ Inference + QR decoding complete. JSON saved at {OUTPUT_JSON}")
