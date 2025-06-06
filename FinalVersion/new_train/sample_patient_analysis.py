import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
import sys

work_dir = sys.argv[1]
mode = sys.argv[2]

###################################
#   input
f = open(work_dir + '/predictions_' + mode + '_.txt','r')
lines = f.readlines()
f.close()

predictions = [list(map(float,l.replace('[','').replace(']','').split())) for l in lines]

f = open(work_dir + '/true_' + mode + '_.txt','r')
lines = f.readlines()
f.close()

gt = [int(l) for l in lines]

f = open(work_dir + '/names_' + mode + '_.txt','r')
sample_names = [l.strip() for l in f.readlines()]
names = list(set([l.replace("('startframe_","").replace(".mp4_.nii.gz',)","").split('_')[1] for l in sample_names]))
f.close()

###################################
# AUC per sample
num_classes = 4
test_auc = roc_auc_score(
    y_true=np.eye(num_classes)[gt],
    y_score=predictions,
    average="macro",
    multi_class="ovr",
)
print('Per sample AUC:', test_auc,'\n')

predicted_labels = np.argmax(np.asarray(predictions),axis=1)
cm = confusion_matrix(gt, predicted_labels, labels=[0, 1, 2, 3])

print(cm,'\n')

# Accuracy
accuracy = accuracy_score(gt, predicted_labels)

# Sensitivity = Recall per class
sensitivity = recall_score(gt, predicted_labels, average=None, labels=[0, 1, 2, 3])

specificity = []
for i in range(len(cm)):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    specificity.append(spec)

# Print results
for i, (sens, spec) in enumerate(zip(sensitivity, specificity)):
    print(f"Class {i} → Sensitivity (Recall): {sens:.2f}, Specificity: {spec:.2f}")

print(f"\nOverall Accuracy: {accuracy:.2f}")

###################################
###################################
###################################

case_labels = []
case_predictions = []
for name in names:
    indices = [i for i in range(len(sample_names)) if name + '.mp4_.nii.gz' in sample_names[i] ]
    case_label = np.unique([gt[i] for i in indices])[0]
    case_prediction = np.mean(np.asarray([predictions[i] for i in indices]),axis=0)
    case_labels.append(case_label)
    case_predictions.append(case_prediction)

###################################
# AUC per patient
print('\n###################################\n')

test_auc = roc_auc_score(
    y_true=np.eye(num_classes)[case_labels],
    y_score=case_predictions,
    average="macro",
    multi_class="ovr",
)
print('Per patient AUC:',test_auc,'\n')

case_predicted_labels = np.argmax(np.asarray(case_predictions),axis=1)

cm = confusion_matrix(case_labels, case_predicted_labels, labels=[0, 1, 2, 3])

print(cm,'\n')

# Accuracy
accuracy = accuracy_score(case_labels, case_predicted_labels)

# Sensitivity = Recall per class
sensitivity = recall_score(case_labels, case_predicted_labels, average=None, labels=[0, 1, 2, 3])

specificity = []
for i in range(len(cm)):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    specificity.append(spec)

# Print results
for i, (sens, spec) in enumerate(zip(sensitivity, specificity)):
    print(f"Class {i} → Sensitivity (Recall): {sens:.2f}, Specificity: {spec:.2f}")

print(f"\nOverall Accuracy: {accuracy:.2f}")

