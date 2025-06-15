import os
import cv2
import csv
from collections import defaultdict

# Paths
VIDEO_PATHS = {
    "broadcast": "data/broadcast.mp4",
    "tacticam": "data/tacticam.mp4"
}
CROPS_PATH = {
    "broadcast": "output/broadcast",
    "tacticam": "output/tacticam"
}
CSV_PATH = "output/player_id_matches.csv"
OUTPUT_DIR = "output/labeled_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV mappings
player_data = defaultdict(list)
with open(CSV_PATH, newline="") as csvfile:
    reader = csv.DictReader(csvfile)  # COMMA-separated, so no delimiter="\t"
    
    # Optional: Clean header and row keys
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    
    for row in reader:
        row = {k.strip(): v for k, v in row.items()}  # clean each row

        player_id = int(row["player_id"])

        t_frame = int(row["t_frame"])
        t_index = int(row["t_index"])
        player_data["tacticam"].append((t_frame, t_index, player_id))

        b_frame = int(row["b_frame"])
        b_index = int(row["b_index"])
        player_data["broadcast"].append((b_frame, b_index, player_id))

# Sort entries by frame number
for cam in player_data:
    player_data[cam].sort()

def draw_player_ids(cam_name):
    print(f"🔄 Annotating: {cam_name}...")
    cap = cv2.VideoCapture(VIDEO_PATHS[cam_name])
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        f"{OUTPUT_DIR}/{cam_name}_annotated.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_map = defaultdict(list)
    for f, i, pid in player_data[cam_name]:
        frame_map[f].append((i, pid))  # i = index

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_map:
            for idx, pid in frame_map[frame_id]:
                crop_file = f"{frame_id}_{idx}.jpg"
                crop_path = os.path.join(CROPS_PATH[cam_name], crop_file)
                if not os.path.exists(crop_path):
                    continue

                img = cv2.imread(crop_path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                # Simulated placement
                x, y = 20 + idx * 30, 50 + idx * 20
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {pid}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"✅ Saved: {OUTPUT_DIR}/{cam_name}_annotated.mp4")

if __name__ == "__main__":
    draw_player_ids("broadcast")
    draw_player_ids("tacticam")
