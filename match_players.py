import pickle
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load extracted features
with open("output/features/broadcast.pkl", "rb") as f:
    B = pickle.load(f)
with open("output/features/tacticam.pkl", "rb") as f:
    T = pickle.load(f)

broadcast_keys = list(B.keys())
broadcast_values = list(B.values())

player_id_map = {}
matches = []
next_player_id = 1

def extract_info(filename):
    frame_id, obj_id = filename.replace(".jpg", "").split("_")
    return int(frame_id), int(obj_id)

for t_fn, t_v in T.items():
    sims = cosine_similarity([t_v], broadcast_values)[0]
    i = np.argmax(sims)
    b_fn = broadcast_keys[i]
    sim_score = sims[i]

    if b_fn not in player_id_map:
        player_id_map[b_fn] = next_player_id
        next_player_id += 1
    player_id = player_id_map[b_fn]

    t_frame, t_idx = extract_info(t_fn)
    b_frame, b_idx = extract_info(b_fn)

    matches.append({
        "player_id": player_id,
        "t_fn": t_fn,
        "b_fn": b_fn,
        "similarity": sim_score,
        "t_frame": t_frame,
        "t_index": t_idx,
        "b_frame": b_frame,
        "b_index": b_idx
    })

# Sort by similarity score
matches.sort(key=lambda x: -x["similarity"])

# Save as CSV
os.makedirs("output", exist_ok=True)
csv_path = "output/player_id_matches.csv"
with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["player_id", "t_fn", "b_fn", "similarity", "t_frame", "t_index", "b_frame", "b_index"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for match in matches:
        writer.writerow(match)

print(f"✅ Player ID matches saved to: {csv_path}")

