# run_gradcam_all.py
import os
import cv2
import torch
import numpy as np
import nibabel as nib
import timm
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import re

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class EchoDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = nib.load(item["img"]).get_fdata().astype(np.float32)
        img = img[:700, 158:858, :]
        if self.transform:
            img = self.transform(img)
        return img, item["label"], item["path"], item["patient_id"]

def extract_patient_id(filename):
    match = re.search(r'_([A-Za-z0-9]+)\.mp4_\.nii\.gz$', filename)
    if match:
        return match.group(1)
    else:
        print(filename)
        return "unknown"

def load_test_data(test_path):
    data = []
    label_map = {"Normal": 0, "HFrEF": 1, "HFpEF": 2, "HFmrEF": 3}
    for class_name in label_map.keys():
        folder = os.path.join(test_path, class_name)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".nii.gz"):
                full_path = os.path.join(folder, file)
                patient_id = extract_patient_id(file)
                data.append({
                    "img": full_path,
                    "label": label_map[class_name],
                    "path": file,
                    "patient_id": patient_id
                })
    return data


def save_heatmap(heatmap, output_path):
    """
    Save the heatmap to a .npy file.
    """
    np.save(output_path, heatmap)

def save_heatmap_jpg(grayscale_cam, input_tensor, save_path_jpg, title=None):
    """
    Save the heatmap as a JPG image with an optional title overlay.
    """
    input_image = input_tensor.squeeze().cpu().numpy()
    if input_image.ndim == 3 and input_image.shape[0] == 30:
        frame_idx = 15
        input_image = input_tensor.squeeze()[frame_idx].cpu().numpy()

    input_image = input_image / np.max(input_image)
    input_image_rgb = np.stack([input_image] * 3, axis=-1)
    cam_image = show_cam_on_image(input_image_rgb, grayscale_cam, use_rgb=True)

    if title:
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)

        lines = title.split(" | ")
        y0, dy = 30, 30
        for i, line in enumerate(lines):
            y = y0 + i * dy
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(cam_image, (10, y - 25), (20 + w, y + 5), bg_color, -1)
            cv2.putText(cam_image, line, (15, y), font, font_scale, text_color, thickness)

    cv2.imwrite(save_path_jpg, cam_image)

def class_name(idx):
    return ["Normal", "HFrEF", "HFpEF", "HFmrEF"][idx]


def get_model(fold, device):
    """
    Load the pre-trained model for the specified fold.
    """
    model = timm.create_model("efficientnet_b3", pretrained=False, in_chans=30, num_classes=4)
    model_path = f"/net/tscratch/people/plgztabor/ECHO/logs1/Best_AUC_fold{fold + 1}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def load_predictions(pred_path, true_path):
    """
    Load predictions and true labels from the specified files.
    """
    with open(pred_path, "r") as f_pred, open(true_path, "r") as f_true:
        predictions = []
        for line in f_pred:
            probs = np.fromstring(line.strip().strip('[]'), sep=' ')
            pred_class = np.argmax(probs)
            predictions.append(int(pred_class))

        truths = [int(line.strip()) for line in f_true]
    return predictions, truths


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = "/net/tscratch/people/plgztabor/ECHO/DATA/Test"
    output_root = "/net/tscratch/people/plggabcza/gradcam_results_2"
    os.makedirs(output_root, exist_ok=True)

    transform = Compose([ToTensor(), Normalize(mean=[0.0], std=[255])])
    test_data = load_test_data(test_path)

    pred_file = "/net/tscratch/people/plgztabor/ECHO/logs1/predictions_test.txt"
    true_file = "/net/tscratch/people/plgztabor/ECHO/logs1/true_test.txt"
    predictions, truths = load_predictions(pred_file, true_file)

    assert len(predictions) == len(test_data), "Mismatch between predictions and test data length"
    assert len(truths) == len(test_data), "Mismatch between true labels and test data length"

    all_results = defaultdict(lambda: defaultdict(list))  # [patient_id][fold] -> list of maps

    for fold in range(5):
        print(f"Processing Fold {fold + 1}")
        model = get_model(fold, device)
        cam = GradCAMPlusPlus(model=model, target_layers=[model.conv_head])

        dataset = EchoDataset(data=test_data, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for idx, (imgs, labels, paths, patient_ids) in enumerate(tqdm(loader)):
            imgs, labels = imgs.to(device), labels.to(device)

            pred_class = predictions[idx]
            true_class = truths[idx]
            correct = pred_class == true_class

            path = paths[0]
            patient_id = patient_ids[0]

            grayscale_cam = cam(input_tensor=imgs, targets=[ClassifierOutputTarget(pred_class)])[0]
            all_results[patient_id][fold].append(grayscale_cam)

            category = "correct" if correct else "incorrect"

            save_dir = os.path.join(output_root, category, str(pred_class))
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, path.replace(".nii.gz", f"_f{fold + 1}.npy"))
            save_heatmap(grayscale_cam, save_path)

            save_path_jpg = save_path.replace(".npy", ".jpg")
            title = f"Predicted: {class_name(pred_class)} | True: {class_name(true_class)}"
            save_heatmap_jpg(grayscale_cam, imgs[0], save_path_jpg, title=title)

    # Agregation on models and patients
    for patient_id, fold_maps in all_results.items():
        combined = []
        for fold, maps in fold_maps.items():
            combined.append(np.mean(maps, axis=0))
        patient_map = np.mean(combined, axis=0)

        idxs = [i for i, d in enumerate(test_data) if d["patient_id"] == patient_id]
        if not idxs:
            continue

        pred_class = predictions[idxs[0]]
        true_class = truths[idxs[0]]

        category = "correct" if pred_class == true_class else "incorrect"

        patient_save_dir = os.path.join(output_root, "aggregated", category, str(pred_class))
        os.makedirs(patient_save_dir, exist_ok=True)

        save_path = os.path.join(patient_save_dir, f"{patient_id}.npy")
        save_heatmap(patient_map, save_path)

        sample_entry = test_data[idxs[0]]
        sample_img = nib.load(sample_entry["img"]).get_fdata().astype(np.float32)
        sample_img = sample_img[:700, 158:858, :]
        sample_img = transform(sample_img).unsqueeze(0).to(device)

        save_path_jpg = save_path.replace(".npy", ".jpg")
        pred_counts = defaultdict(int)
        for idx in idxs:
            pred_counts[predictions[idx]] += 1

        pred_summary = ", ".join(f"{k}:{v}" for k, v in sorted(pred_counts.items()))
        majority_pred = max(pred_counts.items(), key=lambda x: x[1])[0]

        title = (
            f"Aggregated: {len(idxs)} images | "
            f"Predicted Count: {pred_summary} | "
            f"Majority Pred: {class_name(majority_pred)} | "
            f"True: {class_name(true_class)}"
        )

        save_heatmap_jpg(patient_map, sample_img[0], save_path_jpg, title=title)

if __name__ == "__main__":
    main()
