# Aiml-Internship-Project--OpenCv
# ⚽ Multi-Angle Football Player ID Matching 🎥 + 🧠

This project uses **Computer Vision** and **Deep Learning** to detect, track, and match football players across different camera angles — like **Broadcast** and **Tacticam** views — from match footage.

### 🚀 Powered by:
- ⚡ YOLOv8 (for real-time player detection)
- 🧠 ResNet-50 (for deep feature extraction)
- 📏 Cosine Similarity (for visual matching)
- 🏷️ OpenCV (for drawing ID labels on video)


## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🧍‍♂️ Player Detection | Detect players in every frame using YOLOv8 |
| ✂️ Crop Extraction | Save each detected player as a separate image |
| 🧠 Deep Feature Extraction | Use ResNet-50 to get vector embeddings of player crops |
| 🔁 Player Matching | Match players across views using cosine similarity |
| 🆔 Consistent ID Assignment | Assign unique Player IDs for multi-view consistency |
| 🎨 Video Annotation | Overlay bounding boxes & IDs onto video frames |
| 📊 CSV Logging | Save all matches (IDs, similarity, frame indices) to a CSV |

## 🧪 Output Summary

After execution, you’ll get:

📁 `output/player_id_matches.csv` — Matched player pairs with similarity scores  
📂 `output/features/` — Deep feature `.pkl` files  
📂 `output/broadcast/` and `output/tacticam/` — YOLO-detected player crops  
🎬 `output/labeled_videos/broadcast_annotated.mp4`  
🎬 `output/labeled_videos/tacticam_annotated.mp4`

## 🛠️ Environment Setup

### ✅ Requirements

- Python 3.8 or higher
- (Optional) CUDA-enabled GPU for faster YOLO inference

### 📦 Install Dependencies

Create a virtual environment (recommended) and install all packages:

```bash
pip install -r requirements.txt
ultralytics
opencv-python
torch
torchvision
scikit-learn
Pillow

Project Structure
Aiml Internship/
├── data/
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── models/
│   └── best.pt                 # YOLOv8 trained weights
├── output/
│   ├── broadcast/              # YOLO crops
│   ├── tacticam/               # YOLO crops
│   ├── features/               # ResNet embeddings
│   ├── labeled_videos/         # Final annotated videos
│   └── player_id_matches.csv   # Final match result
├── detect_players.py
├── extract_features.py
├── match_players.py
├── match_cross_camera.py
└── draw_ids.py

How to Run the Project
🔍 Step 1: Detect Players and Save Crops
bash
python detect_players.py

Step 2: Extract Deep Features with ResNet
bash
python extract_features.py

Step 3: Match Players Across Angles
bash
python match_players.py

Step 4:  Player Matching Using Cosine Similarity

This script matches player crops from two different camera angles by comparing their feature embeddings using cosine similarity and prints the top 20 closest pairs.

bash
python match_cross_camera.py

Step 5: Draw Player Id's on Original broadcast

This script overlays player IDs on the original broadcast and tacticam videos using bounding boxes based on matched crops and frame indices.
bash
python draw_ids.py







