import sys

import torch
import torch.nn as nn
import timm
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, InterpolationMode
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import nibabel as nib

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = nib.load(item["img"]).get_fdata().astype(np.float32)
        img = img[:700,158:858,:]
        label = item["label"]
        fname = item["fname"]
        if self.transform:
            img = self.transform(img)
        return img, label, fname

def main(logs_base_model, mode):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == 'test':
        dataset_dir = "/net/tscratch/people/plgztabor/ECHO/DATA/Test/"
    else:
        dataset_dir = "/net/tscratch/people/plgztabor/ECHO/DATA/Train/"

    test_transforms = Compose([ToTensor(), Normalize(mean=[0.0], std=[255])])

    models = {
        "EfficientNetB3": timm.create_model("efficientnet_b3", pretrained=True, in_chans=30, num_classes=4).to(device),
    }

    num_folds = 5
    num_classes = 4
    for model_name, model in models.items():

        print(f"============================MODEL {model_name}============================")

        test_data = []
        if mode == 'test':
            for cls_name, label in zip(["Normal", "HFrEF", "HFpEF", "HFmrEF"], [0, 1, 2, 3]):
                test_dir = os.path.join(dataset_dir, f"{cls_name}")
                for file in os.listdir(test_dir):
                    if file.endswith(".nii.gz"):
                        sample = {"img": os.path.join(test_dir, file), "label": label, "fname": file}
                        test_data.append(sample)
        else:
            for i in range(num_folds):
                for cls_name, label in zip(["Normal", "HFrEF", "HFpEF", "HFmrEF"], [0, 1, 2, 3]):
                    fold_dir = os.path.join(dataset_dir, f"Fold{i}/{cls_name}")
                    for file in os.listdir(fold_dir):
                        if file.endswith(".nii.gz"):
                            sample = {"img": os.path.join(fold_dir, file), "label": label, "fname": file}
                            test_data.append(sample)
            

        test_ds = CustomDataset(data=test_data, transform=test_transforms)

        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

        # Collect true labels once
        test_labels = []
        fnames = []
        with torch.no_grad():
            for _, labels, fname in test_loader:
                test_labels.extend(labels.numpy())
                fnames.append(fname)
        test_labels = np.array(test_labels)

        # Collect predictions across folds
        all_fold_predictions = []
        for fold in range(num_folds):
            print(f"Loading best model for Fold {fold + 1}...")
            best_model_path = os.path.join(logs_base_model, f"Best_Loss_fold{fold + 1}.pth") 
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()

            fold_predictions = []
            with torch.no_grad():
                for imgs, _ , _ in test_loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    fold_predictions.append(probabilities)

            fold_predictions = np.concatenate(fold_predictions, axis=0)
            all_fold_predictions.append(fold_predictions)

        # Aggregate and calculate AUC
        all_fold_predictions = np.array(all_fold_predictions)
        ensemble_predictions = np.mean(all_fold_predictions, axis=0)
        test_auc = roc_auc_score(
            y_true=np.eye(num_classes)[test_labels],
            y_score=ensemble_predictions,
            average="macro",
            multi_class="ovr",
        )

        f = open(logs_base_model + '/names_' + mode + '_.txt','w')
        for item in fnames:
            print(item,file=f)
        f.close()

        f = open(logs_base_model + '/true_' + mode + '_.txt','w')
        for item in test_labels:
            print(item,file=f)
        f.close()

        f = open(logs_base_model + '/predictions_' + mode + '_.txt','w')
        for i in range(ensemble_predictions.shape[0]):
            print(ensemble_predictions[i],file=f)
        f.close()
        
        f = open(logs_base_model + '/results_' + mode + '_.txt','w')
        print("=== Test Results ===",file=f)
        print(f"Test Set AUC: {test_auc:.4f}",file=f)
        print(f"Certainty (Standard Deviation) across folds: {np.std(all_fold_predictions, axis=0).mean()}",file=f)
        f.close()

if __name__ == "__main__":

    work_dir = sys.argv[1]
    mode = sys.argv[2]
    main(work_dir, mode)

