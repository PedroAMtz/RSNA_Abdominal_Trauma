import pandas as pd 
import numpy as np
import os
import cv2
import pydicom
import nibabel as nib
from glob import glob
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


def window_converter(image, window_width=400, window_level=50):      
    img_min = window_level - window_width // 2
    img_max = window_level + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    #image = (image / image.max() * 255).astype(np.float64)
    return window_image

def transform_to_hu(medical_image, image):
    meta_image = pydicom.dcmread(medical_image)
    intercept = meta_image.RescaleIntercept
    slope = meta_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    # Correct DICOM pixel_array if PixelRepresentation == 1.
        pixel_array = dcm.pixel_array
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
        return pixel_array

def resize_img(img_paths, target_size=(128, 128)):
        volume_shape = (target_size[0], target_size[1], len(img_paths)) 
        volume = np.zeros(volume_shape, dtype=np.float64)
        for i, image_path in enumerate(img_paths):
            image = pydicom.read_file(image_path)
            image = standardize_pixel_array(image)
            hu_image = transform_to_hu(image_path, image)
            window_image = window_converter(hu_image)
            image = cv2.resize(window_image, target_size)
            volume[:,:,i] = image
        return volume
    
def normalize_volume(resized_volume):
    original_shape = resized_volume.shape
    flattened_image = resized_volume.reshape((-1,))
    scaler = preprocessing.MinMaxScaler()
    normalized_flattened_image = scaler.fit_transform(flattened_image.reshape((-1, 1)))
    normalized_volume_image = normalized_flattened_image.reshape(original_shape)
    return normalized_volume_image

def create_3D_segmentations(filepath, target_size, downsample_rate=1):
    img = nib.load(filepath).get_fdata()
    img = np.transpose(img, [2, 1, 0])
    img = np.rot90(img, -1, (1,2))
    img = img[::-1,:,:]
    img = np.transpose(img, [2, 1, 0])
    img = img[::downsample_rate, ::downsample_rate, ::downsample_rate]
    
    resized_images = []

    for i in range(img.shape[2]):
        resized_img = cv2.resize(img[:, :, i], target_size)
        resized_images.append(resized_img)
    
    resized_3D_mask = np.stack(resized_images, axis=2)
    
    return np.array(resized_3D_mask, dtype=np.int8)

def generate_patient_processed_data(list_img_paths, list_seg_paths, target_size=(128,128)):

    height = target_size[0]
    width = target_size[1]
    depth = len(list_img_paths)

    volume_array = np.zeros((height, width, depth), dtype=np.float64)

    print("Initializing data preprocessing with the following dimensions-> Volumes:{}".format(volume_array.shape))

    resized_images = resize_img(list_img_paths, target_size=target_size)
    normalized_siz_volume = normalize_volume(resized_images)
    volume_array = normalized_siz_volume
    volume_mask = create_3D_segmentations(list_seg_paths, target_size=target_size)
    
    transposed_volume_dcm = np.transpose(volume_array, (2, 0, 1))
    transpose_volume_nii = np.transpose(volume_mask, (2, 0, 1))
    
    labelencoder = LabelEncoder()

    n, h, w = transpose_volume_nii.shape
    train_masks_reshaped = transpose_volume_nii.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped.ravel())
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    transposed_volume_dcm = np.expand_dims(transposed_volume_dcm, axis=3)
    transpose_volume_nii = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    return transposed_volume_dcm, transpose_volume_nii