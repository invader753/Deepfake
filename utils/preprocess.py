import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

try:
    from facenet_pytorch import MTCNN
    _MTCNN = True
except ImportError:
    _MTCNN = False

# DeepfakeBench standard normalization (mapping to [-1, 1])
TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std =[0.5, 0.5, 0.5]
    )
])

def crop_face(image, scale=1.3):
    """
    Detect, align, and crop the primary face using MTCNN.
    Aligns the face 2D horizontally based on eye keypoints to maximize CNN accuracy.
    """
    if not _MTCNN:
        return image

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Suppress output to avoid spamming logs
    mtcnn = MTCNN(keep_all=False, device=device)
    
    # Needs RGB PIL Image
    image = image.convert("RGB")
    
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    if boxes is not None and len(boxes) > 0 and landmarks is not None:
        box = boxes[0]
        pts = landmarks[0]
        
        # 2D Facial Alignment
        # pts[0] is typically the left eye (image left), pts[1] right eye (image right)
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image to level eyes
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        # Re-detect after alignment
        boxes2, probs2 = mtcnn.detect(image)
        if boxes2 is not None and len(boxes2) > 0:
            box = boxes2[0]
            
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        
        cx = xmin + width / 2
        cy = ymin + height / 2
        
        side = max(width, height) * scale
        
        x1 = int(cx - side / 2)
        y1 = int(cy - side / 2)
        x2 = int(cx + side / 2)
        y2 = int(cy + side / 2)
        
        w, h = image.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        arr = np.array(image)
        face = arr[y1:y2, x1:x2]
        if face.size > 0:
            return Image.fromarray(face)

    return image

def preprocess_image(image_bytes):
    """
    Complete preprocessing pipeline: Square Crop -> Resize -> Transform.
    Returns the primary cropped PIL Image (scale=1.3) and a list of normalized tensors at multiple scales.
    """
    import io
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    scales = [1.2, 1.3, 1.4]
    tensors = []
    primary_crop = None
    
    for s in scales:
        cropped_face = crop_face(image, scale=s)
        tensor = TRANSFORM(cropped_face)
        tensors.append(tensor)
        if s == 1.3:
            primary_crop = cropped_face
            
    return primary_crop, tensors
