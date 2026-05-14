import os
import io
import uuid
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from PIL import Image

from utils.preprocess import preprocess_image
from utils.heatmap import generate_heatmap

# ── Xception Architecture ──

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, d=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, k, s, p, d, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.conv1(x))

class Block(nn.Module):
    def __init__(self, in_f, out_f, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = None
        if out_f != in_f or strides != 1:
            self.skip = nn.Conv2d(in_f, out_f, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_f)

        rep = []
        filters = in_f

        if grow_first:
            rep += [nn.ReLU(inplace=True), SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False), nn.BatchNorm2d(out_f)]
            filters = out_f

        for _ in range(reps - 1):
            rep += [nn.ReLU(inplace=True), SeparableConv2d(filters, filters, 3, 1, 1, bias=False), nn.BatchNorm2d(filters)]

        if not grow_first:
            rep += [nn.ReLU(inplace=True), SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False), nn.BatchNorm2d(out_f)]

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        skip = self.skip(inp) if self.skip else inp
        if self.skip:
            skip = self.skipbn(skip)
        return x + skip

class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, False, True)
        self.block2 = Block(128, 256, 2, 2, True, True)
        self.block3 = Block(256, 728, 2, 2, True, True)

        for i in range(4, 12):
            setattr(self, f'block{i}', Block(728, 728, 3, 1, True, True))

        self.block12 = Block(728, 1024, 2, 2, True, False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        for i in range(1, 13):
            x = getattr(self, f'block{i}')(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ── Model Loading ──

def get_efficientnet(num_classes=2):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading models on {device}...")
    
    # Load Xception
    xception_model = Xception()
    xception_path = "model/xception.pth"
    if os.path.exists(xception_path):
        state_dict = torch.load(xception_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            # DeepfakeBench stores weights with 'backbone.' prefix
            if k.startswith('backbone.'):
                k = k.replace('backbone.', '')
            
            # Map pointwise to pw
            if 'pointwise' in k:
                k = k.replace('pointwise', 'pw')
            
            # Map last_linear to our fc layer
            if k == 'last_linear.weight':
                k = 'fc.weight'
            elif k == 'last_linear.bias':
                k = 'fc.bias'
                
            new_state_dict[k] = v
            
        xception_model.load_state_dict(new_state_dict, strict=False)
        print(f"[OK] Loaded Xception weights")
    else:
        print("[!] Warning: model/xception.pth not found, using initialized weights")
    xception_model.to(device).eval()

    # Load EfficientNet-B4
    effnet_model = get_efficientnet()
    effnet_path = "model/efficientnet.pth"
    effnet_loaded = False
    if os.path.exists(effnet_path):
        effnet_model.load_state_dict(torch.load(effnet_path, map_location=device), strict=False)
        print(f"[OK] Loaded EfficientNet weights")
        effnet_loaded = True
    else:
        print("[!] Warning: model/efficientnet.pth not found. EfficientNet will be disabled in ensemble.")
    effnet_model.to(device).eval()

    return {"xception": xception_model, "efficientnet": effnet_model}, device, effnet_loaded

# Initialize models globally
MODELS, DEVICE, EFFNET_LOADED = load_models()

# Calibration Threshold (Default 50.0, tune if false positives/negatives are high)
DETECTION_THRESHOLD = 50.0

# Class Mapping Toggle
# FaceForensics++ usually trains with 0=Real, 1=Fake.
# If your model outputs "REAL" for fake images, change this to True.
INVERT_CLASSES = True

def predict(image_bytes, save_dir="static/heatmaps"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Preprocess (returns a list of tensors for different crop scales)
    cropped_pil, tensors = preprocess_image(image_bytes)
    
    # Test-Time Augmentation (TTA): Multi-Scale + Horizontal Flip
    batch_list = []
    for t in tensors:
        t_orig = t.unsqueeze(0).to(DEVICE)
        t_flip = torch.flip(t_orig, dims=[3])
        batch_list.extend([t_orig, t_flip])
        
    batch_tensors = torch.cat(batch_list, dim=0)

    # Ensemble Inference
    with torch.no_grad():
        logits_xc = MODELS["xception"](batch_tensors)
        probs_xc = torch.softmax(logits_xc, dim=1).mean(dim=0)
        
        if EFFNET_LOADED:
            logits_eff = MODELS["efficientnet"](batch_tensors)
            probs_eff = torch.softmax(logits_eff, dim=1).mean(dim=0)
            
            # Average probabilities across ensemble
            p0 = (probs_xc[0].item() + probs_eff[0].item()) / 2.0
            p1 = (probs_xc[1].item() + probs_eff[1].item()) / 2.0
            model_used_text = "Xception + EfficientNet-B4 Ensemble (Multi-Scale TTA)"
        else:
            p0 = probs_xc[0].item()
            p1 = probs_xc[1].item()
            model_used_text = "Xception (Single Model + Multi-Scale TTA)"

    # Normalize
    total = p0 + p1 + 1e-6
    p0 /= total
    p1 /= total

    # DeepfakeBench typically uses Class 0 = Fake, Class 1 = Real
    real = p1 * 100
    fake = p0 * 100

    if INVERT_CLASSES:
        real, fake = fake, real

    # --- BLUR COMPENSATION ---
    # Deepfake models often mistake natural blur for forgery artifacts.
    # We calculate the Laplacian variance (sharpness) of the cropped face.
    cv_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # If the image is blurred (sharpness < 150), the model is biased toward 'Fake'.
    # We aggressively raise the threshold to stop false positives on low-quality photos.
    dynamic_threshold = DETECTION_THRESHOLD
    if sharpness < 150.0:
        # Raise threshold by up to 5 points depending on blur severity
        blur_penalty = ((150.0 - sharpness) / 150.0) * 5.0
        dynamic_threshold += blur_penalty
        print(f"[*] Blurry image detected (Sharpness: {sharpness:.1f}). Raised threshold to {dynamic_threshold:.1f}")

    label = "FAKE" if fake > dynamic_threshold else "REAL"

    # Generate Heatmap (using Xception by default for stability on the primary scale 1.3 tensor)
    primary_tensor = tensors[1] # Scale 1.3 is the middle one
    heatmap_tensor = primary_tensor.unsqueeze(0).to(DEVICE)
    heatmap_tensor.requires_grad_(True)
    heatmap_pil = generate_heatmap(MODELS["xception"], heatmap_tensor, cropped_pil)
    
    heatmap_filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
    heatmap_path = os.path.join(save_dir, heatmap_filename)
    heatmap_pil.save(heatmap_path, quality=85)

    return {
        "label": label,
        "confidence": round(max(real, fake), 2),
        "real_score": round(real, 2),
        "fake_score": round(fake, 2),
        "heatmap_path": f"/static/heatmaps/{heatmap_filename}",
        "model_used": model_used_text
    }