import os
import nibabel as nib
import numpy as np
import re
import cv2


def extract_frames(video_path: str, crop_coords: tuple) -> list:
    """
    Extracts frames from a video, converts them to grayscale, crops them to the specified region,
    and applies simple threshold-based segmentation to mask the background.

    Parameters:
    video_path (str): Path to the input video file.
    crop_coords (tuple): Coordinates for cropping the frames in the format (x_start, x_end, y_start, y_end).

    Returns:
    list: A list of extracted and segmented grayscale frames as NumPy arrays.
    """
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            raise Exception(f"Nie można otworzyć pliku wideo: {video_path}")

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale and crop it to the region of interest
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cropped_frame = gray_frame[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]

            frames.append(cropped_frame)

        cap.release()
        print(f"Extracted {len(frames)} video frames")
        return frames

    except Exception as e:
        print(f"Error has occurred {e}")
        return []



    except Exception as e:
        print(f"Error has occurred {e}")
        return []


def max_image_nr(output_dir: str):
    """
    Returns the highest image number found in the output directory.

    Parameters:
    output_dir (str): Path to the directory containing the images.

    Returns:
    int: The maximum image number found, or 0 if no images are present.
    """
    existing_files = os.listdir(output_dir)
    image_numbers = [int(re.search(r'image_(\d+).nii.gz', f).group(1)) for f in existing_files if
                     re.search(r'image_(\d+).nii.gz', f)]
    max_image_number = max(image_numbers, default=0)
    return max_image_number

def create_3d_images_from_video(video_path: str,
                                group_size: int = 30,
                                shift: int = 10,
                                crop_coords=(70, 660, 80, 910),
                                output_dir: str = r'/net/tscratch/people/plggabcza/AIECHO/ImagesDataset3D_2'
                                ) -> list:
    """
    Creates 3D images from a video by extracting frames, grouping them, and saving them as NIfTI files.

    Parameters:
    video_path (str): Path to the input video file.
    group_size (int): Number of frames to include in each 3D image. Default is 30.
    shift (int): Number of frames to shift for each new group. Default is 10.
    output_dir (str): Directory where the output NIfTI files will be saved. Default is '/net/tscratch/people/plggabcza/AIECHO/ImagesDataset3D'.

    Returns:
    list: A list of file paths to the created 3D NIfTI images.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from video, that are cropped to the region of interest
    video_frames = extract_frames(video_path,crop_coords=crop_coords)
    num_frames = len(video_frames)

    images = []

    # Iterate through the frames and create 3D images
    for start in range(0, num_frames - group_size + 1, shift):

        group = video_frames[start:start + group_size]
        img_3d = np.stack(group, axis=-1) # Create 3D image from the group of frames
        nifti_image = nib.Nifti1Image(
            img_3d.astype(np.float32), affine=np.eye(4)) #np.eye(4) is a 4x4 identity matrix, no transformations
        file_name = f'image_{max_image_nr(output_dir)+1}.nii.gz' # Create a unique file name
        nib.save(nifti_image, os.path.join(output_dir, file_name))
        images.append(os.path.join(output_dir, file_name))

    return images


if __name__ == '__main__':

    base_dir = r'/net/tscratch/people/plggabcza/AIECHO/Dataset'
    output_base_dir = r'/net/tscratch/people/plggabcza/AIECHO/ImagesDataset3D_2'

    for main_dir in ['Train', 'Test']:
        
        main_dir = os.path.join(base_dir, main_dir)
        
        for subfolder in ['HFpEF', 'HFrEF', 'Normal', 'HFmrEF']:
            
            subfolder_dir = os.path.join(main_dir, subfolder)
            output_subfolder_dir = os.path.join(output_base_dir, main_dir, subfolder)
            os.makedirs(output_subfolder_dir, exist_ok=True)
            
            for file in os.listdir(subfolder_dir):
                
                if file.endswith('.mp4'):
                    
                    video_path = os.path.join(subfolder_dir, file)
                    results = create_3d_images_from_video(video_path, output_dir=output_subfolder_dir)
                    print(f"Finished processing {file} in {main_dir}/{subfolder}")
                    print(f'----------------------------')