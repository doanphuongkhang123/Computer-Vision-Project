"""Single-image demo for VTransAdaptive (PathFlow).

Tiles an image into an NxN grid of patches, encodes each patch with the UNI2
foundation model, classifies the patch sequence, and overlays a per-patch
cross-attention heatmap (prototype -> patch attention) on the original image.

Env vars:
  CONFIG      model/class config yaml (default: configs/bracs_server.yaml)
  CHECKPOINT  trained VTransAdaptive weights (.pth); random weights if unset
  DEVICE      cuda / cpu (auto-detected)

UNI2-h is a gated model: run `huggingface-cli login` (or set HF_TOKEN) first.
"""
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import timm
import cv2
import gradio as gr
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_transformer_model import VTransAdaptive

CONFIG = os.environ.get("CONFIG", "configs/bracs_server.yaml")
CHECKPOINT = os.environ.get("CHECKPOINT", "")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PATCH = 224

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)
CLASS_NAMES = cfg.get("class_names", ["Benign", "Atypical", "Malignant"])

_M = {}


def _load():
    if _M:
        return
    kw = dict(
        img_size=PATCH, patch_size=16, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    enc = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw).eval().to(DEVICE)
    tf = timm.data.create_transform(**timm.data.resolve_model_data_config(enc), is_training=False)

    clf = VTransAdaptive(
        num_classes=len(CLASS_NAMES), ratio=float(cfg["ratio"]),
        dropout=float(cfg["dropout"]), hidden_dim=int(cfg["hidden_size"]),
        cls_ratio=float(cfg["cls_ratio"]), max_retrieved=cfg.get("max_retrieved"),
    )
    if CHECKPOINT and os.path.exists(CHECKPOINT):
        ck = torch.load(CHECKPOINT, map_location="cpu")
        sd = ck.get("state_dict", ck.get("model", ck))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, _ = clf.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint {CHECKPOINT} (missing keys: {len(missing)})")
    else:
        print("WARNING: no CHECKPOINT set -> random classifier weights (predictions not meaningful).")
    clf.eval().to(DEVICE)
    _M.update(enc=enc, tf=tf, clf=clf)


def _tiles(img, g):
    im = Image.fromarray(img).convert("RGB").resize((g * PATCH, g * PATCH))
    return [im.crop((x * PATCH, y * PATCH, x * PATCH + PATCH, y * PATCH + PATCH))
            for y in range(g) for x in range(g)]


@torch.no_grad()
def predict(img, grid):
    if img is None:
        return {}, None
    _load()
    g = int(grid)
    x = torch.stack([_M["tf"](p) for p in _tiles(img, g)]).to(DEVICE)
    feats = torch.cat([_M["enc"](x[i:i + 32]).float() for i in range(0, len(x), 32)]).unsqueeze(0)

    mask = torch.zeros(1, feats.size(1), dtype=torch.bool, device=DEVICE)
    logits, attn, idxs = _M["clf"](feats, mask, epoch=0, stride=1, return_attn=True)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # mean prototype->patch attention, mapped back to original tile order
    imp = torch.zeros(feats.size(1), device=DEVICE)
    imp[idxs] = attn.mean(1)[0]
    imp = imp.cpu().numpy().reshape(g, g)
    imp = (imp - imp.min()) / (np.ptp(imp) + 1e-8)

    h, w = img.shape[:2]
    heat = cv2.resize((imp * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img + 0.45 * heat).astype(np.uint8)

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}, overlay


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="numpy", label="Tissue image"),
            gr.Slider(4, 12, value=8, step=1, label="Patch grid (N x N)")],
    outputs=[gr.Label(num_top_classes=len(CLASS_NAMES), label="Classification + confidence"),
             gr.Image(label="Attention heatmap (where the model looks)")],
    title="PathFlow / VTransAdaptive demo",
    description="Image is tiled into patches, encoded with UNI2, and classified. "
                "The heatmap shows the learnable-prototype cross-attention per patch.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8502)
