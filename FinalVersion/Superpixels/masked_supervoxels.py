import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import slic, find_boundaries

# Wczytanie obrazu NIfTI
fname = '/net/tscratch/people/plggabcza/image_332.nii.gz'
im = nib.load(fname).get_fdata()

# Sprawdzenie wymiarów
print("Wymiary oryginalnego obrazu:", im.shape)  # Powinno być (708, 1016, 30)

# Tworzenie maski - musi mieć ten sam kształt co im
mask = im > 0

# Normalizacja obrazu do zakresu [0, 255]
im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Przekształcenie obrazu do formatu RGB dla SLIC
im_rgb = np.stack([im] * 3, axis=-1)  # (708, 1016, 30, 3)

# Dopasowanie maski do 4-wymiarowego obrazu
mask_rgb = np.stack([mask] * 3, axis=-1)  # (708, 1016, 30, 3)

# Sprawdzenie wymiarów
print("Nowy kształt obrazu:", im_rgb.shape)
print("Nowy kształt maski:", mask_rgb.shape)

# Segmentacja SLIC z poprawioną maską
superpixels = slic(im_rgb, n_segments=500, compactness=5.0, max_num_iter=100, sigma=2,
                   enforce_connectivity=True, slic_zero=False, start_label=1, mask=mask)

# Znalezienie granic segmentów
boundaries = find_boundaries(superpixels, mode='thick').astype(np.uint8) * 255

# Zapis do NIfTI
nib.save(nib.Nifti1Image(superpixels.astype(np.uint16), affine=np.eye(4)), 'masked_superpixels.nii.gz')
nib.save(nib.Nifti1Image(boundaries.astype(np.uint8), affine=np.eye(4)), 'masked_boundaries.nii.gz')

print("✅ Maska zastosowana, pliki zapisane!")
