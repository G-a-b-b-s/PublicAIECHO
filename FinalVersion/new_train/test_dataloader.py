import sys

import torch
import torch.nn as nn
import timm
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, InterpolationMode, GaussianBlur, ColorJitter, RandomApply
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import nibabel as nib
from tqdm import tqdm

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ModifyContrast:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, tensor):
        return (tensor - tensor.min())/(tensor.max()-tensor.min())*self.ratio*tensor.max()

    def __repr__(self):
        return f"{self.__class__.__name__}(ratio={self.ratio})"


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
        if self.transform:
            img = self.transform(img)
        return img, label

def assign_label(subfolder):
    labels = {"HFpEF": 2, "HFrEF": 1, "Normal": 0, "HFmrEF": 3}
    return labels.get(subfolder, -1)


def main(fold_to_run):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = "/net/tscratch/people/plgztabor/ECHO/DATA/"

    class_weights = {0: 1/55, 1:1/165, 2:1/138, 3:1/60}

    #################################
    weight_decay = 3e-5
    max_norm = 5
    drop_rate = 0.3
    batch_size = 8
    ################################


    train_transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(90, InterpolationMode.BILINEAR),
        Normalize(mean=[0.0], std=[255]),
        RandomApply([GaussianBlur(kernel_size = 5,sigma = 10)], p=0.5),
        RandomApply([AddGaussianNoise(0., 0.03),],p=0.25),
        RandomApply([ModifyContrast(ratio=0.5)], p=0.5)
    ])
    val_transforms = Compose([ToTensor(), Normalize(mean=[0.0], std=[255])])


    num_folds = 5
    num_classes = 4
    results = {}


    for fold in range(5):

        if fold != fold_to_run:
            continue

        print(f"Starting Fold {fold + 1}/{num_folds}")

        val_data = []
        train_data = []
        train_weights = []

        for i in range(num_folds):
            for cls_name, label in zip(["Normal", "HFrEF", "HFpEF", "HFmrEF"], [0, 1, 2, 3]):
                fold_dir = os.path.join(dataset_dir, f"Train/Fold{i}/{cls_name}")
                for file in os.listdir(fold_dir):
                    if file.endswith(".nii.gz"):
                        sample = {"img": os.path.join(fold_dir, file), "label": label}
                        weight = class_weights[label]
                        if i == fold_to_run:
                            val_data.append(sample)
                        else:
                            train_data.append(sample)
                            train_weights.append(weight)

        trainSampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
        train_ds = CustomDataset(data=train_data, transform=train_transforms)
        val_ds = CustomDataset(data=val_data, transform=val_transforms)


        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=trainSampler, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        for imgs, labels in tqdm(train_loader):
            print(imgs.shape)


        for imgs, labels in tqdm(val_loader):
            print(imgs.shape)



if __name__ == "__main__":

    fold_to_run = int(sys.argv[1])
    main(fold_to_run)

