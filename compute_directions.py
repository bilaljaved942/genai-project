"""
compute_directions.py
Computes latent direction vectors for face attributes using the trained encoder.

For each attribute (Blond Hair, Smile, Age, Male):
  - Picks 300 positive examples and 300 negative examples from CelebA
  - Encodes them with the trained encoder (encoder only, NO generator needed)
  - direction = mean(W+_positive) - mean(W+_negative)
  - Saves direction as a .pth file

Run with:
    C:\\Users\\BILAL\\miniconda3\\envs\\genai_test\\python.exe compute_directions.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CELEBA_DIR    = os.path.join(SCRIPT_DIR, "archive", "img_align_celeba", "img_align_celeba")
ATTR_CSV      = os.path.join(SCRIPT_DIR, "archive", "list_attr_celeba.csv")
ENCODER_PATH  = os.path.join(SCRIPT_DIR, "encoder_best.pth")
OUTPUT_DIR    = SCRIPT_DIR   # save direction .pth files here

# Attributes to compute — these become sliders in the final app
# Format: (csv_column_name, output_filename, friendly_name)
ATTRIBUTES = [
    ("Blond_Hair",   "d_blond.pth",   "Blond Hair"),
    ("Smiling",      "d_smile.pth",   "Smiling"),
    ("Young",        "d_age.pth",     "Young/Age"),
    ("Male",         "d_male.pth",    "Male/Female"),
    ("Bangs",        "d_bangs.pth",   "Bangs"),
]

SAMPLES_PER_CLASS = 300   # 300 pos + 300 neg per attribute
BATCH_SIZE        = 16    # process 16 images at once (encoder only = fast on CPU)

# ── Imports ────────────────────────────────────────────────────────────────────
print("Importing libraries...")
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ── FaceEncoder ────────────────────────────────────────────────────────────────
class FaceEncoder(nn.Module):
    def __init__(self, w_dim=512, num_ws=18):
        super().__init__()
        self.num_ws = num_ws
        backbone = tv_models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.heads    = nn.ModuleList([nn.Linear(512, w_dim) for _ in range(num_ws)])

    def forward(self, x):
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        return torch.stack([h(feat) for h in self.heads], dim=1)  # (B, 18, 512)

# ── Simple Dataset ─────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.paths     = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# ── Load encoder ───────────────────────────────────────────────────────────────
print(f"\nLoading encoder from {ENCODER_PATH}...")
encoder = FaceEncoder(w_dim=512, num_ws=18).to(DEVICE).eval()
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
print("Encoder loaded.")

# ── Load attribute CSV ─────────────────────────────────────────────────────────
print(f"\nLoading attribute labels from {ATTR_CSV}...")
df = pd.read_csv(ATTR_CSV)
print(f"Loaded: {len(df)} images, {len(df.columns)-1} attributes")
print(f"Columns: {list(df.columns[:5])} ... (and more)")

# ── Helper: encode a list of image paths ──────────────────────────────────────
def encode_images(image_paths):
    """Encode a list of image paths -> W+ tensor (N, 18, 512)"""
    dataset    = FaceDataset(image_paths, transform)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    all_ws     = []
    with torch.no_grad():
        for imgs in loader:
            ws = encoder(imgs.to(DEVICE))  # (B, 18, 512)
            all_ws.append(ws.cpu())
    return torch.cat(all_ws, dim=0)  # (N, 18, 512)

# ── Compute directions ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Computing attribute direction vectors")
print("="*60)

for col, filename, name in ATTRIBUTES:
    if col not in df.columns:
        print(f"\n[SKIP] Column '{col}' not found in CSV.")
        continue

    out_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(out_path):
        print(f"\n[SKIP] {filename} already exists — delete to recompute.")
        continue

    print(f"\n[{name}] ({col})")

    # Pick positive and negative samples
    pos_ids = df[df[col] ==  1]["image_id"].sample(SAMPLES_PER_CLASS, random_state=42).tolist()
    neg_ids = df[df[col] == -1]["image_id"].sample(SAMPLES_PER_CLASS, random_state=42).tolist()

    pos_paths = [os.path.join(CELEBA_DIR, f) for f in pos_ids]
    neg_paths = [os.path.join(CELEBA_DIR, f) for f in neg_ids]

    # Filter to only existing files
    pos_paths = [p for p in pos_paths if os.path.exists(p)]
    neg_paths = [p for p in neg_paths if os.path.exists(p)]
    print(f"  Positive samples: {len(pos_paths)}  |  Negative: {len(neg_paths)}")

    # Encode positive class
    print(f"  Encoding positive ({name}) faces...")
    ws_pos = encode_images(pos_paths)   # (300, 18, 512)
    print(f"  Encoding negative (non-{name}) faces...")
    ws_neg = encode_images(neg_paths)   # (300, 18, 512)

    # Direction = mean difference in W+ space
    d = ws_pos.mean(0) - ws_neg.mean(0)  # (18, 512)

    # Normalize the direction vector (unit norm)
    d = d / d.norm()

    torch.save(d, out_path)
    print(f"  Saved -> {filename}  (shape: {d.shape}, norm: {d.norm():.4f})")

print("\n" + "="*60)
print("All direction vectors computed!")
print("="*60)
print("\nFiles saved:")
for _, filename, name in ATTRIBUTES:
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        d = torch.load(path)
        print(f"  {filename}: shape={d.shape}, norm={d.norm():.4f}  ({name})")

print("\nNext step: build the app using these direction vectors!")
