import logging
import os
import sys
import nibabel as nib
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision.models import vit_b_16
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = nib.load(item["img"]).get_fdata()
        label = item["label"]

        # Ensure img is 3D by selecting the first depth slice
        if img.ndim == 4:
            img = img[..., 0]
        elif img.ndim == 3:
            img = img[:, :, 0]

        # Add a channel dimension if it's missing
        img = np.expand_dims(img, axis=-1)

        img = np.repeat(img, 3, axis=-1)

        img = img.astype(np.float32)
        if self.transform:
            img = self.transform(img)

        return img, label

def assign_label(subfolder):
    """
    Assigns a label based on the subfolder name.

    Args:
        subfolder (str): Name of the subfolder containing images.

    Returns:
        int: Corresponding label for classification.
    """
    labels = {"HFpEF": 2, "HFrEF": 1, "Normal": 0, "HFmrEF": 3}
    if subfolder in labels:
        return labels[subfolder]
    else:
        raise ValueError(f"Unknown class in subfolder: {subfolder}")

def load_dataset(dataset_dir):
    """
    Load dataset from 'Train' and 'Test' directories within the provided dataset directory.
    """
    data = {"Train": [], "Test": []}

    for split in ["Train", "Test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"The directory {split_dir} does not exist.")

        for subfolder in ["HFpEF", "HFrEF", "Normal", "HFmrEF"]:
            subfolder_path = os.path.join(split_dir, subfolder)
            if not os.path.exists(subfolder_path):
                raise ValueError(f"The subfolder {subfolder_path} does not exist in {split_dir}.")

            label = assign_label(subfolder)
            for file in os.listdir(subfolder_path):
                if file.endswith(".nii.gz"):
                    filepath = os.path.join(subfolder_path, file)
                    data[split].append({"img": filepath, "label": label})

    return data["Train"], data["Test"]

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    dataset_dir = "/net/tscratch/people/plggabcza/AIECHO/ImagesDataset3D"
    logs_base_model = "/net/tscratch/people/plggabcza/AIECHO/Training_logs_updated/ViT"
    os.makedirs(logs_base_model, exist_ok=True)
    train_data, test_data = load_dataset(dataset_dir)
    logging.info(f"Number of training samples: {len(train_data)}")
    logging.info(f"Number of test samples: {len(test_data)}")


    random.shuffle(train_data)

    train_transforms = Compose(
        [ToTensor(), Resize((224, 224)), Normalize(mean=[0.5], std=[0.5])]
    )
    val_transforms = Compose(
        [ToTensor(), Resize((224, 224)), Normalize(mean=[0.5], std=[0.5])]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 4
    model = vit_b_16(pretrained=False)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        print(f"Starting Fold {fold + 1}/{num_folds}")

        train_files = [train_data[i] for i in train_idx]
        val_files = [train_data[i] for i in val_idx]

        train_ds = CustomDataset(data=train_files, transform=train_transforms)
        val_ds = CustomDataset(data=val_files, transform=val_transforms)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

        best_auc = 0
        patience = 4
        patience_counter = 0

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
            print(f"Validation AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                model_path = os.path.join(logs_base_model, f"Best_ViT_fold{fold + 1}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved with AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    print("Training completed.")

    test_ds = CustomDataset(data=test_data, transform=val_transforms)
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
        best_model_path = os.path.join(logs_base_model, f"Best_ViT_fold{fold + 1}.pth")
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
if __name__ == "__main__":
    main()
