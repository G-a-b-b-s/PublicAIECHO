import os
import nibabel as nib
import numpy as np
import glob
import cv2

def extract_frames(video_path: str) -> list:
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

            frames.append(gray_frame)

        cap.release()
        print(f"Extracted {len(frames)} video frames")
        return frames

    except Exception as e:
        print(f"Error has occurred {e}")
        return []


def create_3d_images_from_video(video_path: str,
                                output_dir: str,
                                group_size: int = 30,
                                shift: int = 10
                                ) :


    video_frames = extract_frames(video_path)
    num_frames = len(video_frames)
    basename = os.path.basename(video_path)

    for start in range(0, num_frames - group_size + 1, shift):

        group = video_frames[start:start + group_size]
        img_3d = np.stack(group, axis=-1) # Create 3D image from the group of frames
        nifti_image = nib.Nifti1Image(
            img_3d.astype(np.float32), affine=np.eye(4)) #np.eye(4) is a 4x4 identity matrix, no transformations

        file_name = output_dir + '/startframe_' + str(start) + '_' + basename + '_.nii.gz'

        nib.save(nifti_image, os.path.join(output_dir, file_name))


if __name__ == '__main__':

    base_dir = '/net/tscratch/people/plgztabor/ECHO/DATA/'
    dirs = ['Train/Fold' + str(i) + '/' for i in range(5)] + ['Test/',]

    sdirs = [base_dir + 'MP3/' + i for i in dirs]
    ddirs = [base_dir  + i for i in dirs]

    print(sdirs)
    print(ddirs)

    for sdir, ddir in zip(sdirs,ddirs):
        
        for subfolder in ['HFpEF', 'HFrEF', 'Normal', 'HFmrEF']:
           
            video_files = glob.glob(sdir +'/' + subfolder + '/*.mp4')
            print(len(video_files))
            
            for video_file in video_files:
                    
                create_3d_images_from_video(video_file, ddir + subfolder + '/')
