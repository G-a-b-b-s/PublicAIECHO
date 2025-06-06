import logging
import os
import sys
import time

import nibabel as nib
import random
import numpy as np
import scipy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision.models import ViT_B_32_Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.vision_transformer import ConvStemConfig, Encoder
from torchvision.ops import Conv3dNormActivation
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union, Tuple

import torch
from torchvision.utils import _log_api_usage_once


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        # Set target size to (depth, height, width)
        self.target_size = (32, 672, 832)  # New target size

    def __len__(self):
        return len(self.data)

    def resize_volume(self, img, target_size):
        """Resize 3D volume to target dimensions using interpolation."""
        zoom_factors = [t / s for t, s in zip(target_size, img.shape)]
        resized_img = scipy.ndimage.zoom(img, zoom_factors, order=2)
        return resized_img

    def __getitem__(self, idx):
        item = self.data[idx]
        start_time = time.time()
        img = nib.load(item["img"]).get_fdata()
        label = item["label"]
        print(f"Loaded {item['img']} in {time.time() - start_time:.2f} seconds")
        if img.ndim == 4:
            img = np.mean(img, axis=-1)  # Average if 4D

        img = np.transpose(img, (2, 0, 1))

        # Resize to target size (D, H, W)
        img = self.resize_volume(img, self.target_size)

        # Normalize and add channel dimension: (C, D, H, W)
        img = (img - img.mean()) / img.std()
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        if self.transform:
            img = torch.tensor(img)
            img = self.transform(img)
        else:
            img = torch.tensor(img)

        return img, label

class VisionTransformer3D(nn.Module):
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int, int]],  # Can be int or (depth, height, width)
            patch_size: Union[int, Tuple[int, int, int]],  # Can be int or (d, h, w)
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 4,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Convert to tuples if single int provided
        if isinstance(image_size, int):
            image_size = (image_size, image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # 3D patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels=1,  # single channel for medical images
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Calculate number of patches
        self.num_patches = (image_size[0] // patch_size[0]) * \
                           (image_size[1] // patch_size[1]) * \
                           (image_size[2] // patch_size[2])

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer encoder
        self.encoder = Encoder(
            seq_length=self.num_patches + 1,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # Classification head
        self.head = nn.Linear(hidden_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # Patch embedding
        x = self.patch_embed(x)  # (B, hidden_dim, n_d, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)

        # Classification head
        x = x[:, 0]  # Use class token for classification
        x = self.head(x)

        return x

def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer3D:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 850)

    model = VisionTransformer3D(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model
def vit_b_32(*, weights: Optional[ViT_B_32_Weights] = None, progress: bool = True,
             **kwargs: Any) -> VisionTransformer3D:
    weights = ViT_B_32_Weights.verify(weights)
    return _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )
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
    logs_base_model = "/net/tscratch/people/plggabcza/AIECHO/Training_logs_updated/CustomViT"
    os.makedirs(logs_base_model, exist_ok=True)
    train_data, test_data = load_dataset(dataset_dir)
    logging.info(f"Number of training samples: {len(train_data)}")
    logging.info(f"Number of test samples: {len(test_data)}")


    random.shuffle(train_data)

    train_transforms = Compose(
        [ Normalize(mean=[0.5], std=[0.5])]
    )
    val_transforms = Compose(
        [Normalize(mean=[0.5], std=[0.5])]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 4
    model = vit_b_32(image_size=(32, 672, 832))
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
