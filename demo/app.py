"""WSI demo for VTransAdaptive (PathFlow / BRACS).

Opens a whole-slide image (.svs), tiles tissue regions at a target magnification,
encodes each patch with the UNI2 foundation model, classifies the patch sequence,
and overlays a per-patch cross-attention heatmap on the slide thumbnail.

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

os.environ["HF_TOKEN"] = ""
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

CONFIG = os.environ.get("CONFIG", "configs/bracs_server.yaml")
CKPT_REPO = os.environ.get("CKPT_REPO", "tp140205/cv")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
print(f"[demo] device = {DEVICE}")
PATCH = 224
TARGET_MAG = int(os.environ.get("TARGET_MAG", 40))  # must match feature-extraction magnification
SAMPLES = os.environ.get("SAMPLES", "BRACS_1003718.svs,BRACS_1003677_Malignant.svs").split(",")

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
    if not os.environ.get("HF_HUB_OFFLINE"):
        from huggingface_hub import login, whoami
        login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
        print(f"[demo] HuggingFace user: {whoami()['name']}")
    enc = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw).eval().to(DEVICE)
    tf = timm.data.create_transform(**timm.data.resolve_model_data_config(enc), is_training=False)

    clf = VTransAdaptive(
        num_classes=len(CLASS_NAMES), ratio=float(cfg["ratio"]),
        dropout=float(cfg["dropout"]), hidden_dim=int(cfg["hidden_size"]),
        cls_ratio=float(cfg["cls_ratio"]), max_retrieved=cfg.get("max_retrieved"),
    )
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(CKPT_REPO, "best.pth")
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck.get("model", ck))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, _ = clf.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint {ckpt} (missing keys: {len(missing)})")
    clf.eval().to(DEVICE)
    _M.update(enc=enc, tf=tf, clf=clf)


def load_wsi(path, patch=256, target_mag=20, tissue_thr=0.08, max_patches=3000, disp_w=1024):
    """Open a WSI (.svs/.tif/.ndpi), tissue-tile at target_mag, return
    (patches, cells, (rows, cols), display_thumbnail)."""
    from tiffslide import TiffSlide
    sl = TiffSlide(path)
    W, H = sl.dimensions
    base = float(sl.properties.get("openslide.objective-power")
                 or sl.properties.get("aperio.AppMag") or 40)
    ds = max(1.0, base / target_mag)          # level-0 px per target px
    step = max(1, int(round(patch * ds)))     # patch size in level-0 px
    cols, rows = W // step, H // step
    if cols < 1 or rows < 1:
        raise ValueError("Slide smaller than one patch at this magnification.")

    # tissue mask: 1 pixel per patch, high saturation = tissue
    small = np.asarray(sl.get_thumbnail((cols, rows)).convert("RGB"))
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    mask = sat > tissue_thr
    rows, cols = mask.shape
    cells = [(r, c) for r in range(rows) for c in range(cols) if mask[r, c]]
    if not cells:
        raise ValueError("No tissue detected in slide.")
    if len(cells) > max_patches:
        cells = [cells[i] for i in np.linspace(0, len(cells) - 1, max_patches).astype(int)]

    level = sl.get_best_level_for_downsample(ds)
    lvl_ds = sl.level_downsamples[level]
    size_lv = max(1, int(round(patch * ds / lvl_ds)))
    patches = [sl.read_region((c * step, r * step), level, (size_lv, size_lv))
               .convert("RGB").resize((patch, patch)) for (r, c) in cells]
    disp = np.asarray(sl.get_thumbnail((disp_w, int(disp_w * H / W))).convert("RGB"))
    return patches, cells, (rows, cols), disp


@torch.no_grad()
def predict(file):
    if file is None:
        return {}, None
    try:
        path = file.name if hasattr(file, "name") else file
        _load()
        patches, cells, (rows, cols), disp = load_wsi(path, target_mag=TARGET_MAG)

        x = torch.stack([_M["tf"](p) for p in patches]).to(DEVICE)
        feats = torch.cat([_M["enc"](x[i:i + 64]).float() for i in range(0, len(x), 64)]).unsqueeze(0)

        mask = torch.zeros(1, feats.size(1), dtype=torch.bool, device=DEVICE)
        logits, attn, idxs = _M["clf"](feats, mask, epoch=0, stride=1, return_attn=True)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        imp = torch.zeros(feats.size(1), device=DEVICE)
        imp[idxs] = attn.mean(1)[0]
        imp = imp.cpu().numpy()

        # per-patch attention -> slide grid -> heatmap over tissue only
        grid = np.zeros((rows, cols), np.float32)
        for (r, c), v in zip(cells, imp):
            grid[r, c] = v
        tissue = grid > 0
        vals = grid[tissue]
        grid[tissue] = (vals - vals.min()) / (np.ptp(vals) + 1e-8)

        h, w = disp.shape[:2]
        heat = cv2.resize(np.clip(grid, 0, 1), (w, h), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(tissue.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        heat_c = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET),
                              cv2.COLOR_BGR2RGB)
        overlay = (disp * (1 - 0.5 * alpha) + heat_c * (0.5 * alpha)).astype(np.uint8)

        return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}, overlay
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")


with gr.Blocks(title="PathFlow") as demo:
    gr.Markdown(
        "# 🔬 PathFlow — WSI Histopathology Classifier\n"
        "Upload a whole-slide image (**.svs**). Tissue is tiled, encoded with **UNI2**, "
        "and classified by **VTransAdaptive**. The heatmap shows the prototype "
        "cross-attention over the slide — *where the model looks*."
    )
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            inp = gr.File(label="Whole-slide image", file_types=[".svs", ".tif", ".tiff", ".ndpi"])
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            out_label = gr.Label(num_top_classes=len(CLASS_NAMES), label="Diagnosis")
            out_heat = gr.Image(label="Attention heatmap", height=420)

    btn.click(predict, [inp], [out_label, out_heat])

    samples = [s for s in SAMPLES if os.path.exists(s)]
    if samples:
        gr.Examples(
            examples=[[s] for s in samples],
            inputs=[inp], outputs=[out_label, out_heat],
            fn=predict, cache_examples=True,
            label="Preloaded slides (click = instant, no upload)",
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8502, share=True,
                theme=gr.themes.Soft(primary_hue="indigo"))
