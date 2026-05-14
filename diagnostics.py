import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from detect import load_models
from utils.preprocess import crop_face

def test_permutations(image_path):
    print(f"=== DEEPFAKE MODEL DIAGNOSTICS ===")
    print(f"Analyzing: {image_path}\n")
    
    MODELS, DEVICE, EFFNET_LOADED = load_models()
    model = MODELS["xception"]
    model.eval()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {e}")
        return

    # Normalization Strategies
    norm_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm_dfb = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Crop Strategies
    crop_tight = crop_face(img, scale=1.0)
    crop_expanded = crop_face(img, scale=1.3)

    scenarios = [
        ("Tight Crop + ImageNet Norm", crop_tight, norm_imagenet),
        ("Tight Crop + DeepfakeBench Norm", crop_tight, norm_dfb),
        ("Expanded Crop + ImageNet Norm", crop_expanded, norm_imagenet),
        ("Expanded Crop + DeepfakeBench Norm", crop_expanded, norm_dfb),
    ]

    print("\n--- RESULTS ---\n")
    for name, face_img, norm in scenarios:
        # Build transform
        t = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            norm
        ])
        
        tensor = t(face_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
        l0, l1 = logits[0][0].item(), logits[0][1].item()
        p0, p1 = probs[0].item() * 100, probs[1].item() * 100
        
        print(f"Scenario: {name}")
        print(f"  Raw Logits: Class0 = {l0:.3f}, Class1 = {l1:.3f}")
        print(f"  Probs:      Class0 = {p0:.1f}%, Class1 = {p1:.1f}%")
        
        # DeepfakeBench Default (0=Real, 1=Fake)
        pred_dfb = "FAKE" if p1 > p0 else "REAL"
        # Alternate Default (0=Fake, 1=Real)
        pred_alt = "FAKE" if p0 > p1 else "REAL"
        
        print(f"  If mapping is (0=Real, 1=Fake): Outputs {pred_dfb}")
        print(f"  If mapping is (0=Fake, 1=Real): Outputs {pred_alt}")
        print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnostics.py <path_to_image>")
    else:
        test_permutations(sys.argv[1])
