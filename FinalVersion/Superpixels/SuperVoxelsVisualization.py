import nibabel as nib
import matplotlib.pyplot as plt


def visualize_nifti(nifti_path, slice_idx=None):
    nii = nib.load(nifti_path)
    img = nii.get_fdata()

    if slice_idx is None:
        slice_idx = img.shape[-1] // 2

    plt.imshow(img[:, :, slice_idx], cmap='gray')
    plt.title(f'Warstwa {slice_idx}')
    plt.axis('off')
    plt.show()

def compare_segmented(original_nifti, segmented_nifti, slice_idx=None):
    orig = nib.load(original_nifti).get_fdata()
    seg = nib.load(segmented_nifti).get_fdata()

    if slice_idx is None:
        slice_idx = orig.shape[-1] // 2  # Środkowa warstwa

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(orig[:, :, slice_idx], cmap='gray')
    ax[0].set_title("Oryginał")
    ax[0].axis("off")

    ax[1].imshow(seg[:, :, slice_idx], cmap='gray')
    ax[1].set_title("Segmentacja")
    ax[1].axis("off")

    plt.show()


import numpy as np
import imageio
from tqdm import tqdm


def create_gif(nifti_path, output_gif, cmap='gray'):
    nii = nib.load(nifti_path)
    img = nii.get_fdata()

    images = []
    for i in tqdm(range(img.shape[-1])):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img[:, :, i], cmap=cmap)
        ax.axis('off')
        fig.canvas.draw()

        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        images.append(image_array)
        plt.close(fig)

    imageio.mimsave(output_gif, images, duration=0.1)


# visualize_nifti(r"C:\Users\48735\Desktop\Praca\AI_ECHO\Superpixels\superpixels.nii.gz", 2)
# visualize_nifti(r"C:\Users\48735\Desktop\Praca\AI_ECHO\Superpixels\boundaries.nii.gz", 2)
# compare_segmented(r"C:\Users\48735\Desktop\Praca\AI_ECHO\Superpixels\superpixels.nii.gz", r"C:\Users\48735\Desktop\Praca\AI_ECHO\Superpixels\boundaries.nii.gz")
# create_gif(r"/Superpixels/superpixels.nii.gz", r"C:\Users\48735\Desktop\Praca\AI_ECHO\Superpixels\superpixels.gif")
compare_segmented(r"C:\Users\48735\Desktop\AGH\AI_ECHO\Superpixels\image_332.nii.gz", r"C:\Users\48735\Desktop\AGH\AI_ECHO\Superpixels\shap_supervoxels.nii.gz")