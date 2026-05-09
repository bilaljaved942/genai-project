"""
app.py
Interactive Web App for StyleGAN2 Latent Space Manipulation.
"""

import os, sys, warnings, io
warnings.filterwarnings("ignore")

import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
STYLEGAN_DIR = os.path.join(SCRIPT_DIR, "stylegan2-ada-pytorch")
MODEL_PATH   = os.path.join(SCRIPT_DIR, "ffhq.pkl")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "encoder_best.pth")

sys.path.insert(0, STYLEGAN_DIR)

# ── Load StyleGAN2 Generator ───────────────────────────────────────────────────
import dnnlib
import legacy

print("Loading StyleGAN2 Generator...")
with dnnlib.util.open_url(MODEL_PATH) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE).eval().float()
for module in G.modules():
    if hasattr(module, 'use_fp16'):
        module.use_fp16 = False

# ── Define and Load Encoder ────────────────────────────────────────────────────
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
        return torch.stack([h(feat) for h in self.heads], dim=1)

print("Loading FaceEncoder...")
encoder = FaceEncoder(w_dim=G.w_dim, num_ws=G.num_ws).to(DEVICE).eval()
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))

# ── Load Attribute Directions ──────────────────────────────────────────────────
directions = {}
print("Loading Attribute Directions...")
for name, pth in [
    ("Blond Hair", "d_blond.pth"),
    ("Smiling", "d_smile.pth"),
    ("Young", "d_age.pth"),
    ("Male", "d_male.pth"),
    ("Bangs", "d_bangs.pth")
]:
    path = os.path.join(SCRIPT_DIR, pth)
    if os.path.exists(path):
        directions[name] = torch.load(path).to(DEVICE)
        print(f"  Loaded: {name}")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# ── Global State for Editing ───────────────────────────────────────────────────
current_w = None
original_w = None
original_img_display = None

def generate_image(ws):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        img = G.synthesis(ws.float(), noise_mode='const')
    finally:
        sys.stdout = old_stdout
    recon = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
    return (recon[0].permute(1,2,0).cpu().detach().numpy() * 0.5 + 0.5).clip(0, 1)

def process_image(input_image):
    global current_w, original_w, original_img_display
    if input_image is None:
        return None, None
    
    img_pil = Image.fromarray(input_image).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        original_w = encoder(img_tensor)
        current_w = original_w.clone()
        recon_img = generate_image(current_w)
        
    original_img_display = np.array(img_pil.resize((256, 256)))
    return original_img_display, recon_img

def update_attributes(blond, smile, age, male, bangs):
    global current_w, original_w
    if original_w is None:
        return None
    
    w_edit = original_w.clone()
    
    # Apply edits to the first 8 layers for coarse/color attributes
    # Modifying later layers affects fine details and structure too much.
    edit_layers = slice(0, 8) 
    
    if "Blond Hair" in directions:
        w_edit[:, edit_layers, :] += blond * directions["Blond Hair"][edit_layers, :]
    if "Smiling" in directions:
        w_edit[:, edit_layers, :] += smile * directions["Smiling"][edit_layers, :]
    if "Young" in directions:
        w_edit[:, edit_layers, :] += age * directions["Young"][edit_layers, :]
    if "Male" in directions:
        w_edit[:, edit_layers, :] += male * directions["Male"][edit_layers, :]
    if "Bangs" in directions:
        w_edit[:, edit_layers, :] += bangs * directions["Bangs"][edit_layers, :]
        
    current_w = w_edit
    with torch.no_grad():
        recon_img = generate_image(current_w)
    return recon_img

# ── Gradio Web UI ──────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 StyleGAN2 Face Attribute Manipulation")
    gr.Markdown("Upload a photo to map it into the W+ latent space using our trained ResNet18 encoder, and use the sliders to edit attributes!")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Face Photo", type="numpy")
            btn_encode = gr.Button("Encode & Reconstruct", variant="primary")
            
            gr.Markdown("### Attribute Sliders")
            s_blond = gr.Slider(-5, 5, 0, step=0.1, label="Dark Hair ↔ Blond Hair")
            s_smile = gr.Slider(-5, 5, 0, step=0.1, label="Neutral ↔ Smiling")
            s_age = gr.Slider(-5, 5, 0, step=0.1, label="Older ↔ Younger")
            s_male = gr.Slider(-5, 5, 0, step=0.1, label="Female ↔ Male")
            s_bangs = gr.Slider(-5, 5, 0, step=0.1, label="No Bangs ↔ Bangs")
            
            btn_reset = gr.Button("Reset Sliders")
            
        with gr.Column():
            orig_out = gr.Image(label="Original", type="numpy", interactive=False)
            recon_out = gr.Image(label="StyleGAN2 Reconstruction / Edit", type="numpy", interactive=False)

    btn_encode.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[orig_out, recon_out]
    )
    
    sliders = [s_blond, s_smile, s_age, s_male, s_bangs]
    for s in sliders:
        s.change(
            fn=update_attributes,
            inputs=sliders,
            outputs=[recon_out]
        )
        
    def reset_sliders():
        return 0, 0, 0, 0, 0
        
    btn_reset.click(
        fn=reset_sliders,
        inputs=[],
        outputs=sliders
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Launching Web App... Open the Local URL shown below in your browser.")
    print("="*60 + "\n")
    demo.launch(server_name="0.0.0.0", share=False)
