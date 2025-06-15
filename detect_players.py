import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "models/best.pt"
SEGMENTS = {
    "broadcast": "data/broadcast.mp4",
    "tacticam": "data/tacticam.mp4"
}

def detect_and_save(seg, video_path):
    os.makedirs(f"output/{seg}", exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = model(frame)[0]
        for i, box in enumerate(res.boxes.xyxy):
            x1,y1,x2,y2 = map(int, box[:4])
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(f"output/{seg}/{frame_id}_{i}.jpg", crop)
        frame_id += 1
    cap.release()

if __name__ == "__main__":
    for seg, path in SEGMENTS.items():
        detect_and_save(seg, path)
