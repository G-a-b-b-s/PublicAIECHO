import napari
import nibabel as nib
import numpy as np
#nifti_file = r'C:\Users\48735\Desktop\AGH\AI_ECHO\Superpixels\image_332.nii.gz'
#nifti_file = r'C:\Users\48735\Desktop\Praca\AI_ECHO\Superpixels\masked_superpixels.nii.gz'
#nifti_file = r'C:\Users\48735\Desktop\Praca\AI_ECHO\Superpixels\masked_boundaries.nii.gz'
nifti_file = r'C:\Users\48735\Desktop\AGH\AI_ECHO\Superpixels\shap_supervoxels.nii.gz'
nifti_image = nib.load(nifti_file)
data = nifti_image.get_fdata()

viewer = napari.Viewer()

viewer.add_image(data, name='3D Image')

napari.run()
