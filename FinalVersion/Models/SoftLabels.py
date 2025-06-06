import logging
import os
import sys
import random
import seaborn as sns
import monai
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, Normalize
from monai.utils import set_determinism, MAX_SEED
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import nibabel as nib


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List of dictionaries containing image file paths and soft labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = nib.load(item["img"]).get_fdata()
        # Soft labels (e.g., [0.7, 0.1, 0.1, 0.1])
        label = np.array(item["label"], dtype=np.float32)

        # Ensure correct image dimensions
        if img.ndim == 4:
            img = img[..., 0]
        elif img.ndim == 3:
            img = img[:, :, 0]

        # Expand to add channel dimension and repeat to create 3 channels
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)
        img = img.astype(np.float32)

        # If a transform is provided, apply it (note: we convert to a tensor and permute to (C,H,W))
        if self.transform:
            img = self.transform(torch.from_numpy(img).permute(2, 0, 1))
        label = torch.tensor(label, dtype=torch.float32)
        return img, label


def assign_label(subfolder):
    """
    Assigns soft labels based on the subfolder name.
    """
    if subfolder == "HFpEF":  # Heart Failure with Preserved Ejection Fraction
        return [0.15, 0.10, 0.65, 0.10]
    elif subfolder == "HFrEF":  # Heart Failure with Reduced Ejection Fraction
        return [0.10, 0.65, 0.15, 0.10]
    elif subfolder == "HFmrEF":  # Heart Failure with Mid-range Ejection Fraction
        return [0.10, 0.2, 0.05, 0.65]
    elif subfolder == "Normal":  # Healthy heart
        return [0.70, 0.10, 0.10, 0.10]
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
        for subfolder in ["HFpEF", "HFrEF", "HFmrEF", "Normal"]:
            subfolder_path = os.path.join(split_dir, subfolder)
            if not os.path.exists(subfolder_path):
                raise ValueError(f"The subfolder {subfolder_path} does not exist in {split_dir}.")
            label = assign_label(subfolder)
            for file in os.listdir(subfolder_path):
                if file.endswith(".nii.gz"):
                    filepath = os.path.join(subfolder_path, file)
                    data[split].append({"img": filepath, "label": label})
    return data["Train"], data["Test"]


def create_transforms():
    # Note: ToTensor() is omitted because we manually convert to tensor in __getitem__
    train_transforms = Compose([
        Resize((224, 224)),
        Normalize(mean=[0.5], std=[0.5])
    ])
    val_transforms = Compose([
        Resize((224, 224)),
        Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transforms, val_transforms


def prepare_batch(inputs, labels, device=None, non_blocking=False):
    inputs = inputs.to(device, non_blocking=non_blocking)
    labels = labels.to(device, non_blocking=non_blocking)
    return inputs, labels


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    _seed = 42
    _seed %= MAX_SEED
    set_determinism(seed=_seed)

    dataset_dir = "/net/tscratch/people/plggabcza/AIECHO/ImagesDataset3D"
    logs_base_model = "/net/tscratch/people/plggabcza/AIECHO/Training_logs_updated/SoftLabels"
    os.makedirs(logs_base_model, exist_ok=True)

    train_data, test_data = load_dataset(dataset_dir)
    logging.info(f"Number of training samples: {len(train_data)}")
    logging.info(f"Number of test samples: {len(test_data)}")

    random.shuffle(train_data)
    train_transforms, val_transforms = create_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=_seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        print(f"Starting Fold {fold + 1}/{num_folds}")
        train_files = [train_data[i] for i in train_idx]
        val_files = [train_data[i] for i in val_idx]

        train_ds = CustomDataset(train_files, transform=train_transforms)
        val_ds = CustomDataset(val_files, transform=val_transforms)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=4, num_workers=4, pin_memory=True)

        # Using resnet18; modify the first and final layers as needed.
        model = resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        best_auc = 0
        patience = 4
        patience_counter = 0

        for epoch in range(50):
            print(f"Epoch {epoch + 1}/50")
            model.train()
            epoch_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = prepare_batch(inputs, labels, device=device)
                optimizer.zero_grad()
                outputs = model(inputs)
                # Convert soft labels to hard labels for loss computation
                hard_labels = torch.argmax(labels, dim=1)
                loss = criterion(outputs, hard_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")

            print("Validating...")
            model.eval()
            all_val_labels = []
            all_val_preds = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = prepare_batch(inputs, labels, device=device)
                    outputs = model(inputs)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_val_preds.extend(probabilities)
                    all_val_labels.extend(labels.cpu().numpy())

            all_val_labels = np.array(all_val_labels)  # (N, 4) soft labels
            all_val_preds = np.array(all_val_preds)  # (N, 4) predictions
            print(f"val_labels shape: {all_val_labels.shape}, val_preds shape: {all_val_preds.shape}")

            # Convert soft labels to hard labels (via argmax)
            val_labels_hard = np.argmax(all_val_labels, axis=1)
            unique_classes = np.unique(val_labels_hard)
            if len(unique_classes) < 2:
                print("Warning: Only one class present in validation set. ROC AUC is not defined.")
                val_auc = float('nan')
            else:
                # Recreate one-hot encoding from hard labels
                val_labels_one_hot = label_binarize(val_labels_hard, classes=np.arange(4))
                val_auc = roc_auc_score(y_true=val_labels_one_hot, y_score=all_val_preds, average="macro",
                                        multi_class="ovr")
            print(f"Validation AUC: {val_auc:.4f}")

            if not np.isnan(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(logs_base_model, f"Best_SoftLabels_fold{fold + 1}.pth"))
                print(f"New best model saved with AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    print("Evaluating on test set...")
    test_ds = CustomDataset(test_data, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)
    all_predictions = []
    # Compute true test labels from test_data (convert soft labels to hard labels)
    true_test_labels = np.array([np.argmax(item["label"]) for item in test_data])

    for fold in range(num_folds):
        model.load_state_dict(torch.load(os.path.join(logs_base_model, f"Best_SoftLabels_fold{fold + 1}.pth")))
        model.eval()
        fold_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = prepare_batch(inputs, labels, device=device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                fold_predictions.append(probabilities)
        fold_predictions = np.concatenate(fold_predictions, axis=0)
        all_predictions.append(fold_predictions)

    all_predictions = np.array(all_predictions)
    ensemble_predictions = np.mean(all_predictions, axis=0)
    ensemble_std = np.std(all_predictions, axis=0)

    unique_test_labels = np.unique(true_test_labels)
    if len(unique_test_labels) < 2:
        print("Warning: Only one class present in test set. ROC AUC is not defined.")
        test_auc = float('nan')
    else:
        true_test_labels_one_hot = label_binarize(true_test_labels, classes=np.arange(4))
        test_auc = roc_auc_score(y_true=true_test_labels_one_hot, y_score=ensemble_predictions, average="macro",
                                 multi_class="ovr")

    predicted_labels = np.argmax(ensemble_predictions, axis=1)

    # Confusion matrix
    conf_matrix = confusion_matrix(true_test_labels, predicted_labels)
    print("\n=== Confusion Matrix ===")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "HFrEF", "HFpEF", "HFmrEF"],
                yticklabels=["Normal", "HFrEF", "HFpEF", "HFmrEF"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Classification report
    class_report = classification_report(true_test_labels, predicted_labels,
                                         target_names=["Normal", "HFrEF", "HFpEF", "HFmrEF"])
    print("\n=== Classification Report ===")
    print(class_report)

    print("=== Test Results ===")
    print(f"Test Set AUC: {test_auc:.4f}")
    print(f"Certainty (Standard Deviation) across folds: {ensemble_std.mean():.4f}")


if __name__ == "__main__":
    main()
