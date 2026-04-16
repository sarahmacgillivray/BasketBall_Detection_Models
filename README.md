# BasketBall_Detection_Models

## Motivation
 In the world of sports television, multiple camera angles, distracting crowds, and constant movement all vie for the viewer's attention. Basketball, being a fast-paced game, demands a reliable solution to identify the location of the ball on the screen. This can be the foundation of automatic scoring technologies or graphical enhancement technologies for visually impaired viewers. We set our goal to develop a machine learning model that can locate a basketball in any given frame with high emphasis on speed and accuracy. We utilize a dataset of 1,997 training images, 130 validation images and 472 testing images, containing images from various environments, lighting conditions and camera angles. Our baseline is SSD300 with a MobileNetV2 backbone, which achieved 87.04\% mAP@0.5 and 17.90ms/frame. To explore the best model(s) for our project, we conduct experiments on the following options: Faster-RCNN ResNet50, YOLOv8, and a weighted-box-fusion ensemble of combinations of options. Ultimately, YOLOv8 is our final model, achieving a mAP@0.5 score of 92.08\% with an average latency of 6.59ms/frame - fast enough for inferencing on modern video frequencies.

---

## How to Run (YOLOv8n)

This implementation uses separate Python scripts and is designed to run locally.

---

### 1. Install Dependencies

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

Install required packages:

```bash
pip install ultralytics roboflow opencv-python torch torchvision
```

---

### 2. Download Dataset

Run the dataset download script:

```bash
python download_dataset.py
```

This will create a dataset folder (e.g., `basketball-1/`) containing:
- `train/`
- `valid/`
- `test/`
- `data.yaml`

[Dataset] {https://universe.roboflow.com/eagle-eye/basketball-1zhpe/dataset/1/download/yolov8}
---

### 3. Train the Model

Run:

```bash
python train.py
```

The trained model will be saved to:

```
runs/detect/train*/weights/best.pt
```

---

### 4. Evaluate the Model (Test Set)

Run:

```bash
python evaluate.py
```

This will output:
- Precision  
- Recall  
- F1 Score  
- mAP@0.5  
- mAP@0.5:0.95  

Evaluation is performed on the **test split only**.

---

### RNN-ResNet-50
load cpen355-rnn-resnet-50.ipynb on Kaggle with t4 and run all.
### SSD300_VGG16
Load SSD300_VGG16_backbone.ipynb on Kaggle with t4 and run all.
