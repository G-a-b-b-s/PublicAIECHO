import os
import numpy as np
import nibabel as nib
import torch
import lime
import matplotlib.pyplot as plt
from lime import lime_image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from skimage.segmentation import mark_boundaries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(
    torch.load("/net/tscratch/people/plggabcza/AIECHO/Training_logs/ResNet18/Best_ResNet18_fold1.pth"), strict=False)
model.to(device)
model.eval()

image_path = "/net/tscratch/people/plggabcza/image_332.nii.gz"
supervoxel_path = "masked_superpixels.nii.gz"

image_nii = nib.load(image_path)
supervoxel_nii = nib.load(supervoxel_path)

image = image_nii.get_fdata()  # Oryginalny obraz (708, 1016, 30)
supervoxels = supervoxel_nii.get_fdata().astype(int)  # Superwoksele (708, 1016, 30)

transform = Compose([
    ToTensor(),
    Resize((224, 224)),
    Normalize(mean=[0.5], std=[0.5])
])

output_dir = "lime_results"
os.makedirs(output_dir, exist_ok=True)


def predict_fn(images):
    images = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)
    with torch.no_grad():
        preds = model(images).cpu().numpy()
    return preds


for slice_idx in range(image.shape[2])[:2]:
    print(f"Przetwarzanie warstwy {slice_idx + 1}/{image.shape[2]}...")

    image_slice = image[:, :, slice_idx]
    supervoxel_slice = supervoxels[:, :, slice_idx]

    # Maska tła na podstawie czarnych pikseli
    background_mask = image_slice == 0
    if np.all(background_mask):
        continue

    image_slice = np.repeat(image_slice[:, :, np.newaxis], 3, axis=-1).astype(np.float32)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_slice, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
            #masked pixels are set to 0 (hide_color) and top_labels - the most probable class is shown, 1000 samples of image is being testes
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
            # 10 superpixels most important to show, show all image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title(f"Oryginalny obraz - Warstwa {slice_idx + 1}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(f"LIME - Warstwa {slice_idx + 1}")
    plt.axis("off")

    save_path = os.path.join(output_dir, f"lime_visualization_slice_{slice_idx + 1}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

print("Analiza LIME zakończona.")
