from ultralytics import YOLO

def evaluate_model(model_path, yaml_path, class_name="basketball"):
    model = YOLO(model_path)

    metrics = model.val(
        data=yaml_path,
        split="test",
        conf=0.3,
        iou=0.5
    )

    # Get class index for "basketball"
    names = model.names
    class_idx = list(names.values()).index(class_name)

    # Extract per-class metrics
    precision = metrics.box.p[class_idx]
    recall = metrics.box.r[class_idx]
    map50 = metrics.box.ap50[class_idx]
    map5095 = metrics.box.ap[class_idx]

    # Compute F1
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

    print(f"\n=== {class_name.upper()} TEST METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map5095:.4f}")

    return {
        "Class": class_name,
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
