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
    n_epoch = 200
    n_save = 10
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

    models = {
        "EfficientNetB3": timm.create_model("efficientnet_b3", pretrained=True, in_chans=30, num_classes=4, drop_rate=drop_rate).to(device),
    }

    num_folds = 5
    num_classes = 4

    for model_name, model in models.items():

        logs_base_model = "/net/tscratch/people/plgztabor/ECHO/logs2/"
        print(f"============================MODEL {model_name}============================")
        os.makedirs(logs_base_model, exist_ok=True)

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


            best_auc = 0
            best_val_loss = 1e10

            """
            for param in model.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True

            # Unfreeze the last N blocks
            num_blocks_to_unfreeze = 2
            for block in model.blocks[-num_blocks_to_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay = weight_decay)
            """

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = weight_decay)
            
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

            for epoch in range(n_epoch):
                print(f"Epoch {epoch + 1}")
                model.train()
                train_loss = 0.0
                for imgs, labels in tqdm(train_loader):
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)
                    optimizer.step()
                    train_loss += loss.item()

                train_loss = train_loss / len(train_loader)
                print(f"Train Loss: {train_loss:.4f}")

                model.eval()
                val_preds, val_labels = [], []
                val_loss = 0.0
                with torch.no_grad():
                    for imgs, labels in tqdm(val_loader):
                        imgs, labels = imgs.to(device), labels.to(device)
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
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

                val_loss = val_loss / len(val_loader)
                print(f"Validation loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")

                fname =  os.path.join(logs_base_model, f"log_{fold + 1}.txt")
                f = open(fname,'a')
                print(f"{epoch + 1} {train_loss:.4f} {val_loss:.4f} {val_auc:.4f}",file = f) 
                f.close()


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = os.path.join(logs_base_model, f"Best_Loss_fold{fold + 1}.pth")
                    torch.save(model.state_dict(), model_path)
                    print(f"Best loss model saved")

                if  val_auc > best_auc:
                    best_auc = val_auc
                    model_path = os.path.join(logs_base_model, f"Best_AUC_fold{fold + 1}.pth")
                    torch.save(model.state_dict(), model_path)
                    print(f"Best auc model saved")

                if epoch % n_save == 0:
                    model_path = os.path.join(logs_base_model, f"Latest_fold{fold + 1}.pth")
                    torch.save(model.state_dict(), model_path)
                    print(f"Latest model saved")
                    

        print("Training completed.")


if __name__ == "__main__":

    fold_to_run = int(sys.argv[1])
    main(fold_to_run)

