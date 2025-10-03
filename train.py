import os
from ultralytics import YOLO

def main():
    # Paths
    data_yaml = os.path.join("src", "datasets", "data.yaml")   # dataset config
    model_name = "yolov8n.pt"                             # base model

    # Load YOLO model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=100,          # number of training epochs
        imgsz=640,          # image size
        batch=16,           # adjust if GPU memory is low
        name="qr_detector", # experiment name, saved under runs/detect/
        device=0
    )

    # Save the final model
    model_path = os.path.join("outputs", "qr_detector.pt")
    os.makedirs("outputs", exist_ok=True)
    model.save(model_path)

    print(f"\n✅ Training complete. Model saved at: {model_path}")


if __name__ == "__main__":
    main()
