import os
import shutil
import random

def create_folders(base_dir):
    train_dir = os.path.join(base_dir, 'Train')
    test_dir = os.path.join(base_dir, 'Test')
    subfolders = ['HFpEF', 'HFrEF', 'Normal', 'HFmrEF']

    for folder in [train_dir, test_dir]:
        os.makedirs(folder, exist_ok=True)
        for subfolder in subfolders:
            os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    return train_dir, test_dir

def move_files(original_dir, train_dir, test_dir, test_ratio=0.15):
    subfolders = ['HFpEF', 'HFrEF', 'Normal', 'HFmrEF']

    for subfolder in subfolders:
        original_subfolder = os.path.join(original_dir, subfolder)
        train_subfolder = os.path.join(train_dir, subfolder)
        test_subfolder = os.path.join(test_dir, subfolder)

        files = [f for f in os.listdir(original_subfolder) if f.endswith('.mp4')]
        random.shuffle(files)
        test_count = int(len(files) * test_ratio)

        for i, file in enumerate(files):
            src_path = os.path.join(original_subfolder, file)
            if i < test_count:
                dest_path = os.path.join(test_subfolder, file)
            else:
                dest_path = os.path.join(train_subfolder, file)
            shutil.move(src_path, dest_path)

original_dir = '/net/tscratch/people/plggabcza/AIECHO/Data'
base_dir = r'/net/tscratch/people/plggabcza/AIECHO/Dataset'

train_dir, test_dir = create_folders(base_dir)
move_files(original_dir, train_dir, test_dir)