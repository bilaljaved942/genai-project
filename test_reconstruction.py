"""
test_reconstruction.py
Loads encoder_best.pth + StyleGAN2 FFHQ generator and tests reconstruction
on CelebA images (or a fallback synthetic test if not found yet).

Run with:
    C:\\Users\\BILAL\\miniconda3\\envs\\genai_test\\python.exe test_reconstruction.py
"""

import os, sys, glob, random, warnings, io
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
STYLEGAN_DIR = os.path.join(SCRIPT_DIR, "stylegan2-ada-pytorch")
MODEL_PATH   = os.path.join(SCRIPT_DIR, "ffhq.pkl")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "encoder_best.pth")
OUTPUT_IMAGE = os.path.join(SCRIPT_DIR, "reconstruction_result.png")

# ── Find CelebA images — searches common extraction locations ──────────────────
def find_celeba_images(root, n=6):
    """Search for CelebA images anywhere under root directory."""
    candidates = []
    search_patterns = [
        os.path.join(root, "archive", "img_align_celeba", "img_align_celeba", "*.jpg"),
        os.path.join(root, "archive", "img_align_celeba", "*.jpg"),
        os.path.join(root, "img_align_celeba", "img_align_celeba", "*.jpg"),
        os.path.join(root, "img_align_celeba", "*.jpg"),
        os.path.join(root, "celeba", "img_align_celeba", "*.jpg"),
        os.path.join(root, "*.jpg"),
        os.path.join(root, "*.png"),
    ]
    for pattern in search_patterns:
        found = glob.glob(pattern)
        if found:
            print(f"Found {len(found)} image(s) at: {os.path.dirname(pattern)}")
            candidates = found
            break
    if not candidates:
        return []
    random.seed(42)  # reproducible picks
    return random.sample(candidates, min(n, len(candidates)))

sys.path.insert(0, STYLEGAN_DIR)

# ── Verify required files ──────────────────────────────────────────────────────
for path, label in [(MODEL_PATH, "ffhq.pkl"), (ENCODER_PATH, "encoder_best.pth")]:
    if not os.path.exists(path):
        print(f"ERROR: {label} not found at {path}")
        sys.exit(1)
    print(f"{label}: {os.path.getsize(path)/1e6:.1f} MB -- OK")

# ── Pick test images ───────────────────────────────────────────────────────────
celeba_imgs = find_celeba_images(SCRIPT_DIR)

if celeba_imgs:
    print(f"\nUsing {len(celeba_imgs)} CelebA images for testing.")
    USE_CELEBA = True
else:
    print("\nCelebA not found yet (still downloading?). Running synthetic test.")
    print("Re-run this script after extracting the CelebA zip.")
    USE_CELEBA = False

# ── Import torch ───────────────────────────────────────────────────────────────
print("\nImporting torch...")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ── FaceEncoder definition (must exactly match Kaggle training code) ──────────
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
        return torch.stack([h(feat) for h in self.heads], dim=1)  # (B, num_ws, 512)

# ── Load StyleGAN2 Generator ───────────────────────────────────────────────────
print("\nLoading StyleGAN2 FFHQ generator...")
try:
    import dnnlib
    import legacy
    with dnnlib.util.open_url(MODEL_PATH) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE).eval().float()
    # Disable fp16 in all synthesis blocks (required for CPU inference)
    for module in G.modules():
        if hasattr(module, 'use_fp16'):
            module.use_fp16 = False
    print(f"Generator OK: w_dim={G.w_dim}, num_ws={G.num_ws}")
except Exception as e:
    print(f"ERROR loading generator: {e}")
    sys.exit(1)

# ── Load Encoder ───────────────────────────────────────────────────────────────
print("Loading encoder weights...")
encoder = FaceEncoder(w_dim=G.w_dim, num_ws=G.num_ws).to(DEVICE).eval()
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
print("Encoder OK.")

# ── Image transform (same as training) ────────────────────────────────────────
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def to_display(tensor):
    """Convert normalized tensor to displayable numpy array."""
    return (tensor.permute(1,2,0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)

# ── Generate function ──────────────────────────────────────────────────────────
def generate_from_w(ws):
    """ws: (B, num_ws, 512) -> (B, 3, 256, 256)"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        img = G.synthesis(ws.float(), noise_mode='const')
    finally:
        sys.stdout = old_stdout
    return F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)

# ── Run Reconstruction Test ────────────────────────────────────────────────────
if USE_CELEBA:
    n = len(celeba_imgs)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    print(f"\nRunning reconstruction on {n} faces (CPU - this may take 1-3 mins)...")

    for i, img_path in enumerate(celeba_imgs):
        img_name = os.path.basename(img_path)
        print(f"  [{i+1}/{n}] {img_name}...")

        # Load and preprocess
        img_pil    = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        # Encode -> Reconstruct
        with torch.no_grad():
            ws    = encoder(img_tensor)
            recon = generate_from_w(ws)

        # Plot original
        axes[0, i].imshow(img_pil.resize((256, 256)))
        axes[0, i].set_title(f"Original\n{img_name}", fontsize=7)
        axes[0, i].axis("off")

        # Plot reconstruction
        axes[1, i].imshow(to_display(recon[0]))
        axes[1, i].set_title("Reconstructed", fontsize=7)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10, rotation=90, labelpad=10)

else:
    # Synthetic test — encode random noise, should produce a face
    print("Running synthetic test (random noise -> encoder -> generator)...")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    img_tensor = torch.randn(1, 3, 256, 256).to(DEVICE)

    with torch.no_grad():
        ws    = encoder(img_tensor)
        recon = generate_from_w(ws)
        print(f"W+ stats: mean={ws.mean():.4f}, std={ws.std():.4f}")

    noise_display = (img_tensor[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    axes[0].imshow(noise_display); axes[0].set_title("Input (random noise)"); axes[0].axis("off")
    axes[1].imshow(to_display(recon[0])); axes[1].set_title("Encoder output"); axes[1].axis("off")
    fig.suptitle("Synthetic test — extract real face to see real reconstruction", fontsize=10)

# ── Save output ────────────────────────────────────────────────────────────────
plt.suptitle("StyleGAN2 Encoder Reconstruction Quality Test", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nDone! Result saved -> {OUTPUT_IMAGE}")
print("Open reconstruction_result.png to see the quality.")
