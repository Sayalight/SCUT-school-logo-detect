from ultralytics import YOLO
import torch

if __name__ == "__main__":
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train] torch={torch.__version__}  device={device}")
    model = YOLO("yolov8n.pt")
    model.train(
        data="logo.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        device=device,
        project="run/detect",
        name="school_logo",
        patience=50,
        exist_ok=True
    )
