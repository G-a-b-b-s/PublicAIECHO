import os

import shap
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch.nn.functional as F

# === Wczytanie modelu ResNet18 ===
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

resize_transform = Resize((224, 224))


# === Funkcja maskująca superwoksele ===
def mask_supervoxel(image_tensor, supervoxel_slice, label, fill_value=0):
    """
    Maskuje dany superwoksel na obrazie 3D. Wyłączamy dany superwoksel , przekształcamy slice by pasował do modelu
    :param image_tensor:
    :param supervoxel_slice:
    :param label:
    :param fill_value:
    :return:
    """
    supervoxel_slice_resized = torch.tensor(supervoxel_slice).unsqueeze(0).unsqueeze(0).float()
    supervoxel_slice_resized = F.interpolate(supervoxel_slice_resized, size=(224, 224),
                                             mode="nearest").squeeze().cpu().numpy()
    masked_image = image_tensor.clone()
    masked_image[:, :, supervoxel_slice_resized == label] = fill_value
    return masked_image

output_dir = "shap_masked_background"
os.makedirs(output_dir, exist_ok=True)

# === Obliczanie SHAP dla superwokseli na całym obrazie 3D ===
shap_values_3d = np.zeros_like(supervoxels, dtype=np.float32)

for slice_idx in range(image.shape[2]):  # Iterujemy po każdej warstwie 2D
    print(f"Przetwarzanie warstwy {slice_idx + 1}/{image.shape[2]}...")

    image_slice = image[:, :, slice_idx]
    supervoxel_slice = supervoxels[:, :, slice_idx]

    # Maksa tła na podstawie czarnych pikseli
    background_mask = image_slice == 0

    # Jeśli cała warstwa to tło, pomijamy ją
    if np.all(background_mask):
        continue

    image_slice = np.repeat(image_slice[:, :, np.newaxis], 3, axis=-1).astype(np.float32)
    image_tensor = transform(image_slice).unsqueeze(0).to(device)

    unique_supervoxels = np.unique(supervoxel_slice[~background_mask])  # Ignorujemy tło
    baseline_pred = model(image_tensor).cpu().detach().numpy()

    for label in unique_supervoxels:
        masked_tensor = mask_supervoxel(image_tensor, supervoxel_slice, label)
        masked_pred = model(masked_tensor).cpu().detach().numpy()
        shap_values_3d[supervoxel_slice == label, slice_idx] = np.abs(baseline_pred - masked_pred).sum()

    # === Wizualizacja SHAP dla tej warstwy ===
    shap_slice = shap_values_3d[:, :, slice_idx]
    shap_slice[background_mask] = 0  # Wyzerowanie wartości SHAP dla tła

    if shap_slice.max() > 0:
        shap_normalized = (shap_slice - shap_slice.min()) / (shap_slice.max() - shap_slice.min())
    else:
        shap_normalized = shap_slice

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, slice_idx], cmap='gray')
    plt.title(f"Oryginalny obraz - Warstwa {slice_idx + 1}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image[:, :, slice_idx], cmap='gray', alpha=0.5)
    plt.imshow(shap_normalized, cmap='jet', alpha=0.7)
    plt.colorbar(label="SHAP Importance Score")
    plt.title(f"Mapa SHAP - Warstwa {slice_idx + 1}")
    plt.axis("off")

    save_path = os.path.join(output_dir, f"shap_visualization_slice_{slice_idx + 1}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# Wyzerowanie szap dla tła
shap_values_3d[image == 0] = 0

# === Zapis wyników SHAP do pliku NIfTI ===
shap_nifti = nib.Nifti1Image(shap_values_3d, affine=image_nii.affine)
shap_filename = os.path.join(output_dir, "shap_supervoxels.nii.gz")
nib.save(shap_nifti, shap_filename)

print(f" Analiza SHAP zakończona. ")