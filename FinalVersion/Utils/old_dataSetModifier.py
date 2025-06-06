import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Ustaw ścieżkę do katalogu z danymi
base_dir = Path(r"/net/tscratch/people/plggabcza/AIECHO/ImagesDataset3D/Train")
classes = ["Normal", "HFpEF", "HFrEF", "HFmrEF"]
num_folds = 5

for fold in range(num_folds):
    for cls in classes:
        fold_dir = base_dir / f"Fold{fold}" / cls
        fold_dir.mkdir(parents=True, exist_ok=True)

class_files = {cls: list((base_dir / cls).iterdir()) for cls in classes}

# Wymieszanie danych i podział na foldy
folds = {i: defaultdict(list) for i in range(num_folds)}

for cls in classes:
    files = class_files[cls]
    random.shuffle(files)  # losowe przemieszanie danych
    fold_size = len(files) // num_folds

    for i in range(num_folds):
        start = i * fold_size
        # ostatni fold może mieć więcej, żeby wyrównać
        end = None if i == num_folds - 1 else (i + 1) * fold_size
        folds[i][cls] = files[start:end]

# Przenoszenie plików do odpowiednich folderów Fold*/Class
for fold_idx, fold_data in folds.items():
    for cls, files in fold_data.items():
        for f in files:
            target_path = base_dir / f"Fold{fold_idx}" / cls / f.name
            shutil.copy2(f, target_path)  # lub .move jeśli chcesz przenosić

print("✅ Dane zostały rozdzielone do foldów z zachowaniem proporcji klas.")