from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="basketball-1/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu"  # or "cuda" once GPU works
    )

if __name__ == "__main__":
    main()