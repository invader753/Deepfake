import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        target = output[0][target_class]
        target.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

def apply_colormap_on_image(org_im, activation, colormap_name=cv2.COLORMAP_JET):
    """
    Apply a heatmap onto an image.
    org_im: PIL Image
    activation: 2D numpy array [0, 1]
    """
    org_im_arr = np.array(org_im)
    if len(org_im_arr.shape) == 2:
        org_im_arr = cv2.cvtColor(org_im_arr, cv2.COLOR_GRAY2RGB)
    
    # Resize activation to match image
    activation = cv2.resize(activation, (org_im_arr.shape[1], org_im_arr.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), colormap_name)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend
    res = cv2.addWeighted(org_im_arr, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(res)

def generate_heatmap(model, input_tensor, original_image, target_layer_name="auto"):
    """
    Tries to generate a Grad-CAM heatmap.
    If it fails (e.g., target layer not found), returns a dummy heatmap overlay.
    """
    try:
        # Auto-detect target layer based on model type
        target_layer = None
        if hasattr(model, 'conv4'): # Xception
            target_layer = model.conv4
        elif hasattr(model, 'features'): # EfficientNet
            target_layer = model.features[-1]
        else:
            return original_image # Fallback

        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate(input_tensor)
        heatmap_img = apply_colormap_on_image(original_image, cam)
        return heatmap_img
    except Exception as e:
        print(f"[!] Heatmap generation failed: {e}")
        return original_image
