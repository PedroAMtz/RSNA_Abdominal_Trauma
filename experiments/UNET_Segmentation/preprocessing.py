import pandas as pd 
import numpy as np
import os
import cv2
import pydicom
import nibabel as nib
from glob import glob
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import sqlite3


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

    return volume_array, volume_mask

def compute_class_weights_and_encode_masks(volume_segmentations):
  labelencoder = LabelEncoder()  
  n, h, w = volume_segmentations.shape
  train_masks_reshaped = volume_segmentations.reshape(-1,1)
  train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped.ravel())
  train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
  number_classes = len(np.unique(train_masks_reshaped_encoded))
  class_weights = class_weight.compute_class_weight('balanced',
                                                 classes = np.unique(train_masks_reshaped_encoded),
                                                 y = train_masks_reshaped_encoded)

  return train_masks_encoded_original_shape, number_classes, class_weights

def transpose_and_expand_data(volume_images, volume_masks_encoded):
  transposed_volume_dcm = np.transpose(volume_images, (2, 0, 1))
  transpose_volume_nii = np.transpose(volume_masks_encoded, (2, 0, 1))

  transposed_volume_dcm = np.expand_dims(transposed_volume_dcm, axis=3)
  transpose_volume_nii = np.expand_dims(transpose_volume_nii, axis=3)

  print(f"Final data shape: {transposed_volume_dcm.shape}, {transpose_volume_nii.shape}")

  return transposed_volume_dcm, transpose_volume_nii

def generate_data_volumes(data, idx):
  volume_dcm = []
  volume_nii = []

  for i in range(idx):
    volume_img, volume_seg = generate_patient_processed_data(data["patient_paths"][i], data["patient_segmentation"][i])
    
    volume_dcm.append(volume_img)
    volume_nii.append(volume_seg)
  
  volume_of_imgs = np.concatenate(volume_dcm, axis=2)
  volume_of_segs = np.concatenate(volume_nii, axis=2)

  return volume_of_imgs, volume_of_segs

def string_to_list(string_repr):
    return eval(string_repr)

if __name__ == "__main__":

    connection = sqlite3.connect("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/training_data.db")
    sql = pd.read_sql_query("SELECT * FROM segmentations_data", connection)
    cleaned_data = pd.DataFrame(sql, columns =["patient_id","series_id", "patient_paths", "patient_segmentation"])
    cleaned_data["patient_paths"] = cleaned_data["patient_paths"].apply(string_to_list)

    volume_of_imgs, volume_of_segs = generate_data_volumes(data=cleaned_data, idx=50)

    encoded_masks, num_classes, weights = compute_class_weights_and_encode_masks(volume_of_segs)
    print(weights)
    volume_images_cleaned, volume_segs_cleaned = transpose_and_expand_data(volume_images=volume_of_imgs, volume_masks_encoded=encoded_masks)
    
    with open(f'D:/Downloads/rsna-2023-abdominal-trauma-detection/train_data_segmentations/X_y_segmentations_data.npy', 'wb') as f:
        np.save(f, volume_images_cleaned)
        np.save(f, volume_segs_cleaned)