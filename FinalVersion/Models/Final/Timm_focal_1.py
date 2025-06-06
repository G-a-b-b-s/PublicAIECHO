import logging
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
        try:
            img = nib.load(item["img"]).get_fdata().astype(np.float32)
        except Exception as e:
            print(f"Uszkodzony plik pominięty: {item['img']} — {e}")
            return torch.zeros((30, 256, 256)), -1  # lub inny rozmiar pasujący do modelu
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def assign_label(subfolder):
    labels = {"HFpEF": 2, "HFrEF": 1, "Normal": 0, "HFmrEF": 3}
    return labels.get(subfolder, -1)

def load_dataset(dataset_dir, class_weights):
    train_data, test_data = [], []
    train_weights, test_weights = [], []

    # Load TEST
    for cls_name, label in zip(["Normal", "HFrEF", "HFpEF", "HFmrEF"], [0, 1, 2, 3]):
        class_path = os.path.join(dataset_dir, f"Test/{cls_name}")
        if not os.path.exists(class_path):
            continue
        for file in os.listdir(class_path):
            if file.endswith(".nii.gz"):
                sample = {"img": os.path.join(class_path, file), "label": label}
                test_data.append(sample)
                test_weights.append(class_weights[label])
    return train_data, test_data, train_weights, test_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = torch.nn.functional.log_softmax(inputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).float()
        probs = torch.exp(logp)
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        loss = -focal_weight * logp * targets_one_hot
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = "/net/tscratch/people/plggabcza/AIECHO/ImagesDataset3D"

    class_weights = {0: 1/331, 1:1/1370, 2:1/1268, 3:1/506}
    _, test_data, _, _ = load_dataset(dataset_dir, class_weights)

    train_transforms = Compose([
        ToTensor(),
        Pad((0, 120)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(90, InterpolationMode.BILINEAR),
        Normalize(mean=[0.0], std=[255])
    ])
    test_transforms = Compose([ToTensor(), Pad((0, 120)), Normalize(mean=[0.0], std=[255])])

    models = {
        "DenseNet161": timm.create_model("densenet161", pretrained=True, in_chans=30, num_classes=4).to(device),
        "ResNet50d": timm.create_model("resnet50d", pretrained=True, in_chans=30, num_classes=4).to(device),
        "EfficientNetB0": timm.create_model("efficientnet_b0", pretrained=True, in_chans=30, num_classes=4).to(device),
        "ResNet18": timm.create_model("resnet18", pretrained=True, in_chans=30, num_classes=4).to(device),
        # "EfficientNetB3": timm.create_model("efficientnet_b3", pretrained=True, in_chans=30, num_classes=4).to(device),
        # "EfficentNetB4": timm.create_model("efficientnet_b4", pretrained=True, in_chans=30, num_classes=4).to(device),
        # "ResNet34": timm.create_model("resnet34", pretrained=True, in_chans=30, num_classes=4).to(device),
        # "ResNet101": timm.create_model("resnet101", pretrained=True, in_chans=30, num_classes=4).to(device),
        # "InceptionV3": timm.create_model("inception_v3", pretrained=True, in_chans=30, num_classes=4).to(device),
    }

    num_folds = 5
    num_classes = 4
    results = {}
    for model_name, model in models.items():
        logs_base_model = f"/net/tscratch/people/plggabcza/AIECHO/Training_logs_updated/Final/{model_name}_focal"
        print(f"============================MODEL {model_name}============================")
        os.makedirs(logs_base_model, exist_ok=True)

        for fold in range(num_folds):
            print(f"Starting Fold {fold + 1}/{num_folds}")

            # Proper train/val split for this fold
            val_data = []
            train_data = []
            train_weights = []

            for i in range(num_folds):
                for cls_name, label in zip(["Normal", "HFrEF", "HFpEF", "HFmrEF"], [0, 1, 2, 3]):
                    fold_dir = os.path.join(dataset_dir, f"Train/Fold{i}/{cls_name}")
                    if not os.path.exists(fold_dir):
                        continue
                    for file in os.listdir(fold_dir):
                        if file.endswith(".nii.gz"):
                            sample = {"img": os.path.join(fold_dir, file), "label": label}
                            weight = class_weights[label]
                            if i == fold:
                                val_data.append(sample)
                            else:
                                train_data.append(sample)
                                train_weights.append(weight)

            trainSampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
            train_ds = CustomDataset(data=train_data, transform=train_transforms)
            val_ds = CustomDataset(data=val_data, transform=test_transforms)

            print(len(train_ds), len(val_ds))
            print(len(test_data))

            train_loader = DataLoader(train_ds, batch_size=8, sampler=trainSampler, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

            best_auc = 0
            patience = 4
            patience_counter = 0

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

            criterion = FocalLoss(alpha=1, gamma=2)
            for epoch in range(50):
                print(f"Epoch {epoch + 1}")
                model.train()
                train_loss = 0.0
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                print(f"Train Loss: {train_loss / len(train_loader):.4f}")

                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        outputs = model(imgs)
                        probs = torch.softmax(outputs, dim=1)
                        val_preds.append(probs.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                val_preds = np.concatenate(val_preds, axis=0)
                val_labels = np.array(val_labels)
                val_auc = roc_auc_score(
                    y_true=np.eye(num_classes)[val_labels],
                    y_score=val_preds,
                    average="macro",
                    multi_class="ovr",
                )
                scheduler.step(val_auc)
                print(f"Validation AUC: {val_auc:.4f}")

                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0
                    model_path = os.path.join(logs_base_model, f"Best_{model_name}_fold{fold + 1}.pth")
                    torch.save(model.state_dict(), model_path)
                    print(f"Best model saved with AUC: {best_auc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break

        print("Training completed.")

        test_ds = CustomDataset(data=test_data, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

        # Collect true labels once
        test_labels = []
        with torch.no_grad():
            for _, labels in test_loader:
                test_labels.extend(labels.numpy())
        test_labels = np.array(test_labels)

        # Collect predictions across folds
        all_fold_predictions = []
        for fold in range(num_folds):
            print(f"Loading best model for Fold {fold + 1}...")
            best_model_path = os.path.join(logs_base_model, f"Best_{model_name}_fold{fold + 1}.pth")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()

            fold_predictions = []
            with torch.no_grad():
                for imgs, _ in test_loader:
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

        print("=== Test Results ===")
        print(f"Test Set AUC: {test_auc:.4f}")
        print(f"Certainty (Standard Deviation) across folds: {np.std(all_fold_predictions, axis=0).mean()}")
        results[model_name] = {"AUC": test_auc, "Certainty": np.std(all_fold_predictions, axis=0).mean()}
    print(results)
if __name__ == "__main__":
    main()
