import os
import sys
import torch
import torch.nn.utils.prune as prune
import torch.utils.data.dataloader
import subprocess
torch.cuda.empty_cache()
print("[System] Applying Single-Process Patch...", flush=True)
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

_OriginalDataLoader = torch.utils.data.dataloader.DataLoader

class SafeDataLoader(_OriginalDataLoader):
    def __init__(self, *args, **kwargs):
        # Force single-process loading
        kwargs['num_workers'] = 0
        kwargs['persistent_workers'] = False
        super().__init__(*args, **kwargs)

# Overwrite the class globally
torch.utils.data.DataLoader = SafeDataLoader
torch.utils.data.dataloader.DataLoader = SafeDataLoader
print("[System] Patch Applied Successfully.", flush=True)
# -----------------------------------

import cv2
import timm
import numpy as np
import gc
import tempfile
import traceback
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms as T
from werkzeug.utils import secure_filename

# --- 2. PATH SETUP (Fixing Imports) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../FE
PARENT_DIR = os.path.dirname(CURRENT_DIR)                # .../Project_Root
UTILS_DIR = os.path.join(PARENT_DIR, 'utils')            # .../Project_Root/utils

# Add to system path so we can import 'utils' and 'models'
sys.path.append(PARENT_DIR)
sys.path.append(UTILS_DIR)

try:
    from utils.feat_extract import TimmViTEncoder
    from models.vit_transformer_model import VTransAdaptive
    from utils.khang.OF_frames_folder import get_smart_frames_from_list
    print("[System] Project modules imported successfully.", flush=True)
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}", flush=True)
    if os.path.exists(UTILS_DIR):
        print(f"Contents of {UTILS_DIR}: {os.listdir(UTILS_DIR)}")
    sys.exit(1)

# --- 3. CONFIGURATION ---
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 5000 * 1024 * 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/disk4/video-panninng-classification/results/final/baseline_ratio0.1_seed512.pth")
CLASS_NAMES = {0: "Normal", 1: "Adenoma", 2: "Malignant"}
MODELS = {}

# --- 4. MODEL LOADING ---
def load_models_if_needed():
    if "encoder" in MODELS and "classifier" in MODELS:
        return

    print(f"[{os.getpid()}] Loading models on {DEVICE}...", flush=True)
    
    timm_kwargs = {
        'pretrained': True, 'img_size': 224, 'patch_size': 16, 'depth': 24,
        'num_heads': 24, 'init_values': 1e-5, 'embed_dim': 1536,
        'mlp_ratio': 2.66667*2, 'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }

    try:
        # Load Encoder
        encoder = TimmViTEncoder(model_name='hf-hub:MahmoodLab/UNI2-h', kwargs=timm_kwargs)
        encoder.to(DEVICE)
        encoder.eval()
        MODELS["encoder"] = encoder
        
        # Load Classifier
        classifier = VTransAdaptive(
            num_classes=len(CLASS_NAMES), ratio=0.1, dropout=0.5, 
            hidden_dim=1536, n_masked_patch=10, mask_drop=0.1
        )
        
        if os.path.exists(MODEL_PATH):
            print(f"Loading checkpoint: {MODEL_PATH}", flush=True)
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            classifier.load_state_dict(state_dict, strict=False)
            classifier.to(DEVICE)
            for name, module in classifier.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    # This removes the "masks" and makes the pruning permanent for inference
                    prune.remove(module, 'weight')
            
            classifier.eval()
            MODELS["classifier"] = classifier
        else:
            print(f"WARNING: Checkpoint NOT found at {MODEL_PATH}", flush=True)
            # Raise error in production, or remove this line to test with random weights
            raise FileNotFoundError(f"Checkpoint missing: {MODEL_PATH}")
        # encoder.half()  # Convert weights to 16-bit
        # classifier.half()
        print("Models loaded successfully.", flush=True)
    except Exception as e:
        print(f"Model Load Error: {e}", flush=True)
        raise RuntimeError(f"Encoder Load Failed: {e}")
def video_to_frame_list(video_path, scale_factor=0.7, sample_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    width = int(orig_width * scale_factor)
    height = int(orig_height * scale_factor)

    command = [
        'ffmpeg',
        '-i', video_path,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-loglevel', 'error',
        '-'
    ]

    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        bufsize=10**7
    )

    frames = []
    frame_size = width * height * 3 
    count = 0

    while True:
        # Inline strict byte reading to prevent partial pipe reads
        data = bytearray()
        while len(data) < frame_size:
            packet = process.stdout.read(frame_size - len(data))
            if not packet:
                break
            data.extend(packet)
            
        in_bytes = bytes(data)

        if len(in_bytes) != frame_size:
            break
            
        if count % sample_rate == 0:
            frame = np.frombuffer(in_bytes, dtype='uint8')
            frame = frame.reshape((height, width, 3))
            frames.append(frame.copy())
            
        count += 1

    process.stdout.close()
    process.wait()
    
    return frames


# def preprocess_video(video_path):
#     all_frames = video_to_frame_list(video_path)
#     logging.info(f"DEBUG: Completed extraction. Total: {len(all_frames)}")
#     indices = get_smart_frames_from_list(all_frames, 10, 2, 3, 0.3, 0.4)
    
#     selected_images = [all_frames[int(i)] for i in indices]    
#     del all_frames
#     gc.collect()    
#     return torch.stack([T.ToTensor()(img) for img in selected_images]).to(DEVICE)

def preprocess_video(video_path):
    all_frames = video_to_frame_list(video_path)
    logging.info(f"DEBUG: Completed extraction. Total: {len(all_frames)}")
    
    raw_indices = get_smart_frames_from_list(all_frames, 10, 2, 3, 0.3, 0.4)
    
    # Bulletproof flattening: Safely unpacks tuples, 2D arrays, or tensors 
    # and converts them into a flat list of standard Python integers.
    clean_indices = np.array(raw_indices).flatten().astype(int).tolist()
    
    # Now this will work flawlessly
    selected_images = [all_frames[i] for i in clean_indices]    
    
    del all_frames
    gc.collect()    
    return torch.stack([T.ToTensor()(img) for img in selected_images]).to(DEVICE)



# --- 6. ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'device': DEVICE}), 200
        
@app.route('/predict', methods=['POST'])
def predict():
    temp_path = None
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        load_models_if_needed()
        
        f = request.files['video']
        
        # Use a high-speed directory if available (like /dev/shm or a mapped SSD volume)
        fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir="/app") 
        os.close(fd)
        f.save(temp_path)
        
        # 1. Process Video -> List -> Smart Indices -> Tensor
        input_tensor = preprocess_video(temp_path)
        # input_tensor = input_tensor.half()
        
        # 2. Chunked Inference for Encoder
        chunk_size = 25
        all_feats = []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # Process frames in batches of 50 to prevent memory spikes
                for i in range(0, input_tensor.size(0), chunk_size):
                    chunk = input_tensor[i : i + chunk_size]
                    
                    # Encoder (UNI2-h) processing
                    chunk_feat = MODELS["encoder"](chunk).float()
                    all_feats.append(chunk_feat)
                    
                    # Clear intermediate activation memory immediately
                    torch.cuda.empty_cache() 
                
                # Combine all extracted features
                feats = torch.cat(all_feats, dim=0)
                
                # Critical Cleanup: Remove the original large input tensor
                del input_tensor, all_feats
                gc.collect()
                torch.cuda.empty_cache()

            # 3. Classifier Inference
            # Classifier (VTransAdaptive) expects [Batch, Time, Features]
            feats = feats.unsqueeze(0) 
            attn_mask = torch.zeros((1, feats.size(1)), dtype=torch.bool).to(DEVICE)
            
            out = MODELS["classifier"](feats, attn_mask, epoch=1, stride=1)
            probs = torch.softmax(out, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            
            # Final Cleanup
            del feats, attn_mask
            torch.cuda.empty_cache()
            
        return jsonify({
            'predicted_class': CLASS_NAMES.get(idx, str(idx)),
            'confidence': f"{probs[0][idx].item():.2%}",
            'probabilities': {CLASS_NAMES.get(i, i): f"{p:.2%}" for i, p in enumerate(probs[0].tolist())}
        })
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()
        torch.cuda.empty_cache()
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        
# @app.route('/predict', methods=['POST'])
# def predict():
#     temp_path = None
#     try:
#         if 'video' not in request.files: return jsonify({'error': 'No file'}), 400
        
#         load_models_if_needed()
        
#         f = request.files['video']
#         if f.filename == '': return jsonify({'error': 'No filename'}), 400

#         # Save Temp File
#         fd, temp_path = tempfile.mkstemp(suffix=secure_filename(f.filename))
#         os.close(fd)
#         f.save(temp_path)
        
#         # Inference
#         raw = preprocess_video(temp_path)
#         with torch.no_grad():
#             with torch.amp.autocast('cuda' if DEVICE == 'cuda' else 'cpu'):
#                 feats = MODELS["encoder"](raw).float()
            
#             feats = feats.unsqueeze(0)
#             T = feats.size(1)
#             attn_mask = torch.zeros((1, T), dtype=torch.bool).to(DEVICE)
            
#             out = MODELS["classifier"](feats, attn_mask, epoch=1, stride=1)
#             probs = torch.softmax(out, dim=1)
#             idx = torch.argmax(probs, dim=1).item()
            
#         return jsonify({
#             'predicted_class': CLASS_NAMES.get(idx, str(idx)),
#             'confidence': f"{probs[0][idx].item():.2%}",
#             'probabilities': {CLASS_NAMES.get(i, i): f"{p:.2%}" for i, p in enumerate(probs[0].tolist())}
#         })
            
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500
#     finally:
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)
#         if DEVICE == 'cuda':
#             torch.cuda.empty_cache()

if __name__ == '__main__':
    # use_reloader=False is REQUIRED to prevent forking/CUDA crashes
    app.run(host='0.0.0.0', port=8502, debug=False, use_reloader=False)
    
    
    
    
    # @app.route('/convert_preview', methods=['POST'])
# def convert_preview():
#     temp_dir = tempfile.mkdtemp()
#     input_path = None
#     output_path = os.path.join(temp_dir, "preview.mp4")
    
#     try:
#         if 'video' not in request.files:
#             return jsonify({'error': 'No file'}), 400
        
#         f = request.files['video']
#         input_path = os.path.join(temp_dir, secure_filename(f.filename))
#         f.save(input_path)

#         # FFmpeg Command:
#         # -i: input file
#         # -vf scale=-2:480: scale to 480p height, keeping aspect ratio (must be even)
#         # -vcodec libx264: use web-friendly H.264
#         # -crf 28: compress quality (28 is high compression, small size)
#         # -preset veryfast: convert quickly
#         # -y: overwrite output
#         cmd = [
#             'ffmpeg', '-i', input_path,
#             '-vf', 'scale=-2:480',
#             '-vcodec', 'libx264',
#             '-crf', '28',
#             '-preset', 'veryfast',
#             '-movflags', 'frag_keyframe+empty_moov', # Optimized for streaming
#             '-f', 'mp4',
#             output_path, '-y'
#         ]
        
#         subprocess.run(cmd, check=True, capture_output=True)

#         with open(output_path, "rb") as video_file:
#             encoded_video = base64.b64encode(video_file.read()).decode('utf-8')

#         return jsonify({'video_data': encoded_video})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#     finally:
#         # Cleanup all temp files
#         if input_path and os.path.exists(input_path): os.remove(input_path)
#         if os.path.exists(output_path): os.remove(output_path)
#         if os.path.exists(temp_dir): os.rmdir(temp_dir)


# --- 5. PREPROCESSING ---
# def preprocess_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     all_frames = []
    
#     # Get properties for logging
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"[Process] Extracting {total_frames} frames from {video_path}", flush=True)

#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Convert BGR to RGB (Required for your PIL/Transformer workflow)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Optimization: Store as PIL Image or keep as Numpy uint8
#             # PIL Images are slightly more memory efficient than raw float tensors
#             all_frames.append(Image.fromarray(frame))
            
#             # Safety Check: If RAM usage is too high, you can add a break here
#             # if len(all_frames) > 5000: break 
            
#     finally:
#         cap.release()
    
#     if not all_frames:
#         raise ValueError("Extraction failed: No frames found.")

#     # 1. Get smart indices using your utility
#     print(f"[Process] Running get_smart_indices_from_frames on {len(all_frames)} frames...", flush=True)
#     indices = get_smart_indices_from_frames(all_frames)
    
#     # 2. Select the specific frames
#     selected_images = [all_frames[i] for i in indices]
    
#     del all_frames
#     gc.collect() 
#     if DEVICE == "cuda":
#         torch.cuda.empty_cache()

#     # 4. Transform to Tensor for UNI2-h/VTrans
#     transform = T.Compose([T.ToTensor()])
#     video_tensor = torch.stack([transform(img) for img in selected_images]).to(DEVICE)
    
#     return video_tensor
