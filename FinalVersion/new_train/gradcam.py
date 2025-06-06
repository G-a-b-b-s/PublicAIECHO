#pip install grad-cam
# https://github.com/jacobgil/pytorch-grad-cam

import torch
import timm
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Load your trained model (replace with your model name and weights path)
model_name = "resnet50"  # Example TIMM model
model = timm.create_model(model_name, pretrained=True)
#model.load_state_dict(torch.load("path_to_your_trained_model.pth"))
model.eval()

# Choose the target layer for GradCAM++ (usually the last convolutional layer)
target_layers = [model.layer4[-1]]  # Example for ResNet

# Load and preprocess the input image
image_path = "image.jpeg"
img = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Change size if needed for your model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)  # Create a batch

# Run GradCAM++
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(0)]  # Replace 0 with the target class index if known

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
rgb_img = np.array(img.resize((224, 224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Save or display the result
cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


