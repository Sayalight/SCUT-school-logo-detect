import cv2
import time
from ultralytics import YOLO

# best.pt 实际路径（原路径 runs/detect/school_logo/... 不存在）
MODEL_PATH = "runs/detect/run/detect/school_logo/weights/best.pt"
CAM_ID = 0
CONF = 0.5
IOU = 0.45
IMGSZ = 640

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取失败")
        break

    results = model.predict(
        source=frame,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        verbose=False
    )
    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val = float(box.conf[0])
            cls_id = int(box.cls[0])
            names = model.names
            name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            label = f"{name} {conf_val:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 3, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("School Logo Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
