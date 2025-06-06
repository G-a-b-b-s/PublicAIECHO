import os
import shutil
import random
import glob
import numpy as np

data_dir = '/net/tscratch/people/plgztabor/ECHO/DATA/'

subfolders = ['HFpEF', 'HFrEF', 'Normal', 'HFmrEF']

cases = {'HFpEF':[], 'HFrEF':[], 'Normal':[], 'HFmrEF':[]}

TRAIN_TEST_FRACTION = 0.15
NUM_OF_FOLDS = 5

for subfolder in subfolders:
    cases[subfolder] = [os.path.basename(f) for f in glob.glob(data_dir + '/ORIGINAL/' + subfolder + '/*.mp4')]
    np.random.shuffle(cases[subfolder])


for subfolder in subfolders:

    N = int(len(cases[subfolder])*TRAIN_TEST_FRACTION)
    for item in cases[subfolder][:N]:
        sname = data_dir + '/ORIGINAL/' + subfolder + '/' + item
        dname = data_dir + '/Test/' + subfolder + '/' + item
        shutil.copy(sname,dname)

    TRAINING_CASES = int(len(cases[subfolder]) - N)
    CASES_PER_FOLD = TRAINING_CASES / NUM_OF_FOLDS

    SPLIT_POINTS_START = [N + int(i*CASES_PER_FOLD) for i in range(5)]
    SPLIT_POINTS_END = [N + int(i*CASES_PER_FOLD) for i in range(1,5)]
    SPLIT_POINTS_END.append(len(cases[subfolder]))

    for nfold in range(5):
        for item in cases[subfolder][SPLIT_POINTS_START[nfold]:SPLIT_POINTS_END[nfold]]:
            sname = data_dir + '/ORIGINAL/' + subfolder + '/' + item
            dname = data_dir + '/Train/Fold' + str(nfold) + '/' + subfolder + '/' + item
            shutil.copy(sname,dname)
        
