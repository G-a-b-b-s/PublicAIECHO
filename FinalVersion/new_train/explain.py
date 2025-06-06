import sys
import nibabel as nib

import torch
import torch.nn as nn
import timm
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize 
import torch.nn.functional as F

from skimage.transform import resize

work_dir = './logs1/' 
mode = 'test'        

model_paths = ['./logs1/Best_AUC_fold1.pth','./logs1/Best_AUC_fold2.pth','./logs1/Best_AUC_fold3.pth','./logs1/Best_AUC_fold4.pth','./logs1/Best_AUC_fold5.pth']
data_dir = './DATA/Test/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("efficientnet_b3", pretrained=True, in_chans=30, num_classes=4).to(device)

test_transforms = Compose([ToTensor(), Normalize(mean=[0.0], std=[255])])

###################################
#   input
f = open(work_dir + '/predictions_' + mode + '.txt','r')
lines = f.readlines()
f.close()

predictions = [list(map(float,l.replace('[','').replace(']','').split())) for l in lines]

f = open(work_dir + '/true_' + mode + '.txt','r')
lines = f.readlines()
f.close()

gt = [int(l) for l in lines]

f = open(work_dir + '/names_' + mode + '.txt','r')
sample_names = [l.strip() for l in f.readlines()]
names = list(set([l.replace("('startframe_","").replace(".mp4_.nii.gz',)","").split('_')[1] for l in sample_names]))
f.close()
###################################

def save_gradient(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def save_activation(module, input, output):
    activations.append(output)

target_layer = model.conv_head

# Register hooks BEFORE the forward pass
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)  # Use register_full_backward_hook (PyTorch ≥ 1.9+)

class_names = {0:"Normal",1:"HFrEF",2:"HFpEF",3:"HFmrEF"}

for name in names:
    indices = [i for i in range(len(sample_names)) if name + '.mp4_.nii.gz' in sample_names[i] ]
    gt_class = np.unique([gt[i] for i in indices])[0]
    case_prediction = np.mean(np.asarray([predictions[i] for i in indices]),axis=0)
    target_class = np.argmax(case_prediction)

    heatmaps = []
    for index in indices:
        fname = data_dir + class_names[gt_class] + '/' + sample_names[index].replace("('","").replace("',)","")

        #print(fname)
        img = nib.load(fname).get_fdata().astype(np.float32)
        img = img[:700,158:858,:]
        img = test_transforms(img)
        img = img.unsqueeze(axis=0)
        img = img.to(device)
        img.requires_grad_()

        for model_path in model_paths:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            gradients = []
            activations = []

            output = model(img)  # Shape: [1, num_classes]

            # Define target class index

            target = output[0, target_class]

            # Backward pass
            model.zero_grad()
            target.backward()

            # Now retrieve the saved activations and gradients
            grads_val = gradients[0].detach()[0]     # Shape: [C, H, W]
            acts_val = activations[0].detach()[0]    # Shape: [C, H, W]

            # Global average pooling of gradients
            pooled_grads = torch.mean(grads_val, dim=(1, 2))  # Shape: [C]

            for i in range(pooled_grads.shape[0]):
                acts_val[i, :, :] *= pooled_grads[i]

            # Create the heatmap by averaging the channels
            heatmap = torch.mean(acts_val, dim=0)
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap)

            np_heatmap = heatmap.detach().cpu().numpy()

            heatmaps.append(np_heatmap)

    heatmaps = np.mean(np.asarray(heatmaps,dtype=np.float32),axis=0)
    rescaled = resize(heatmaps, img.shape[2:], anti_aliasing=True, preserve_range=True)

    im_slice = img[0,29].detach().cpu().numpy()

    fname = './EXPLANATIONS/' + str(gt_class) + '/heatmap_predicted_' + str(target_class) + '_' + name + '.nii.gz'
    niftiImage = nib.Nifti1Image(rescaled, affine=np.eye(4))
    nib.save(niftiImage,fname)
    
    fname = './EXPLANATIONS/' + str(gt_class) + '/slice_predicted_' + str(target_class) + '_' + name + '.nii.gz'
    niftiImage = nib.Nifti1Image(im_slice, affine=np.eye(4))
    nib.save(niftiImage,fname)

