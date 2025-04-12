import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def generate_gradcam(model, image, target_class):
    model.eval()
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    
    final_conv = list(model.features.children())[-1]
    forward_handle = final_conv.register_forward_hook(forward_hook)
    backward_handle = final_conv.register_backward_hook(backward_hook)

    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    
    grads_val = gradients[0].detach().numpy()[0]
    fmap = features[0].detach().numpy()[0]
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam = np.uint8(255 * cam)

   
    orig_img = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    
    forward_handle.remove()
    backward_handle.remove()

    return overlay
