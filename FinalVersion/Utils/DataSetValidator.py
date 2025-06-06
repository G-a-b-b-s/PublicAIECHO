import os
from collections import defaultdict

dataset_dir = "/net/tscratch/people/plggabcza/AIECHO/ImagesDataset3D"
folds_path = os.path.join(dataset_dir, "Train")
test_path = os.path.join(dataset_dir, "Test")
classes = ["Normal", "HFpEF", "HFrEF", "HFmrEF"]
num_folds = 5

# Step 1: Count files in each fold/class
fold_counts = defaultdict(lambda: defaultdict(int))
fold_files = defaultdict(set)
class_totals = defaultdict(int)

print("🔍 Fold-wise Train File Counts:\n")
for fold in range(num_folds):
    for cls in classes:
        cls_path = os.path.join(folds_path, f"Fold{fold}", cls)
        if not os.path.exists(cls_path):
            print(f"⚠️  Missing folder: {cls_path}")
            continue
        files = [f for f in os.listdir(cls_path) if f.endswith(".nii.gz")]
        fold_counts[f"Fold{fold}"][cls] = len(files)
        fold_files[f"Fold{fold}"].update(files)
        class_totals[cls] += len(files)  # Track total per class
        print(f"Fold{fold} - {cls}: {len(files)} files")

# ✅ Print overall class totals in train set
print("\n🧮 Total per class in all training folds:")
for cls in classes:
    print(f"{cls}: {class_totals[cls]} files")

# Step 2: Count files in Test/class
print("\n📦 Test File Counts:\n")
test_files = set()
test_class_counts = {}

for cls in classes:
    cls_path = os.path.join(test_path, cls)
    if not os.path.exists(cls_path):
        print(f"⚠️  Missing folder: {cls_path}")
        continue
    files = [f for f in os.listdir(cls_path) if f.endswith(".nii.gz")]
    test_class_counts[cls] = len(files)
    test_files.update(files)
    print(f"Test - {cls}: {len(files)} files")
