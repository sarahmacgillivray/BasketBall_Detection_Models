from ultralytics import YOLO

def evaluate_model(model_path, yaml_path):
    model = YOLO(model_path)

    metrics = model.val(
        data=yaml_path,
        split="test",
        conf=0.3,
        iou=0.5
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

    print("\n=== FINAL TEST METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map5095:.4f}")

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095
    }


if __name__ == "__main__":
    evaluate_model(
        "runs/detect/train3/weights/best.pt",
        "basketball-1/data.yaml"
    )