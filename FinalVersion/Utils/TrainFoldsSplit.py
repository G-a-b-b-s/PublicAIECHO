import os
import shutil
import random
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

base_train_dir = r'/net/tscratch/people/plggabcza/AIECHO/Dataset/Train'
output_dir = r'/net/tscratch/people/plggabcza/AIECHO/Dataset/FoldedTrain'
os.makedirs(output_dir, exist_ok=True)

# Wczytaj wszystkie ścieżki i odpowiadające im klasy
video_paths = []
labels = []

for label in ['Normal', 'HFpEF', 'HFrEF', 'HFmrEF']:
    label_dir = os.path.join(base_train_dir, label)
    for fname in os.listdir(label_dir):
        if fname.endswith('.mp4'):
            video_paths.append(os.path.join(label_dir, fname))
            labels.append(label)

#One movie-one patient
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (_, val_idx) in enumerate(skf.split(video_paths, labels)):
    for i in val_idx:
        label = labels[i]
        src_path = video_paths[i]
        dst_dir = os.path.join(output_dir, f'Fold{fold_idx}', label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src_path, os.path.join(dst_dir, os.path.basename(src_path)))

print("Podział danych na foldy zakończony.")
