import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device).eval()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer

# Image preprocessing
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extractor function
def extract(folder):
    feats = {}
    files = os.listdir(folder)
    print(f"\nExtracting features from {len(files)} images in '{folder}'...")

    for idx, fn in enumerate(files, 1):
        path = os.path.join(folder, fn)
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = tfms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                v = resnet(img_tensor).squeeze().cpu().numpy()
                feats[fn] = v / np.linalg.norm(v)

            print(f"[{idx}/{len(files)}] Processed: {fn}")
        except Exception as e:
            print(f"[{idx}/{len(files)}] ❌ Failed to process {fn}: {e}")

    return feats

# Main execution
if __name__ == "__main__":
    import pickle

    os.makedirs("output/features", exist_ok=True)
    for seg in ["broadcast", "tacticam"]:
        seg_path = f"output/{seg}"
        if not os.path.exists(seg_path):
            print(f"⚠️ Skipping missing folder: {seg_path}")
            continue

        feats = extract(seg_path)
        with open(f"output/features/{seg}.pkl", "wb") as f:
            pickle.dump(feats, f)

        print(f"✅ Saved features to output/features/{seg}.pkl")

