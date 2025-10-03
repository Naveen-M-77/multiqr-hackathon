# MultiQR Hackathon Project

This repository contains the code for training, detecting, and decoding multiple QR codes in images using YOLOv8.

## Repository Structure

```
multiqr-hackathon/
│
├── README.md                # Setup & usage instructions
├── requirements.txt         # Python dependencies
├── train.py                 # Training script
├── infer.py                 # Inference for detection
├── infer2.py                # Inference for decoding/classification (bonus)
│
├── data/                    # Placeholder for dataset
│   └── demo_images/         # Demo images for quick testing
│
├── outputs/                 
│   ├── submission_detection_1.json   # Detection results
│   └── submission_decoding_2.json    # Decoding/classification results (bonus)
│
└── src/                     # Source code
    ├── models/
    ├── datasets/
    ├── utils/
    └── __init__.py
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Naveen-M-77/multiqr-hackathon.git
   cd multiqr-hackathon
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Training

Run the training script:

```bash
python train.py
```

* Model will be saved in `outputs/qr_detector.pt`.

## Inference

### Detection

```bash
python infer.py
```

* This runs detection on images in `data/demo_images/`.
* Results (bounding boxes) will be saved as `outputs/submission_detection_1.json`.

### Decoding & Classification (Bonus)

```bash
python infer2.py
```

* This script uses `outputs/qr_detector.pt` for detection and additionally decodes QR values.
* Results will be saved as `outputs/submission_decoding_2.json`.

## Notes

* Ensure `data/demo_images/` contains sample images for testing.
* `src/datasets/data.yaml` should correctly point to your training dataset directories.
