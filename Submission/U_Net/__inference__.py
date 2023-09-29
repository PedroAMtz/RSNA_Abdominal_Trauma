import os
import pandas as pd
import numpy as np
from glob import glob
import tensorflow as tf
import math
import random
import sqlite3
import pydicom
from sklearn import preprocessing
import re
import cv2
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model

""" 
Script for generating dataset considering the predictions made by 
the pretrained U-Net model, this dataset reduction then is converted
into a uniform volume, the final shape is (64, 128, 128, 1)
"""
def window_converter(image, window_width: int=400, window_level: int=50) -> np.ndarray:

    """_Uses the window values in order to create desired contrast to the image_

        Returns
        -------
        _np.ndarray_
            _returns a numpy array with the desired window level applied_
    """
    img_min = window_level - window_width // 2
    img_max = window_level + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def transform_to_hu(medical_image: str, image: np.ndarray) -> np.ndarray:

    """_Tranforms Hounsfield Units considering
            an input image and image path for reading
            metadata_

        Returns
        -------
        _np.ndarray_
            _Returns a numpy array_
    """
    meta_image = pydicom.dcmread(medical_image)
    intercept = meta_image.RescaleIntercept
    slope = meta_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """_Correct DICOM pixel_array if PixelRepresentation == 1._

        Returns
        -------
        _np.ndarray_
            _returns the pixel array from the dicom file with the
            fixed pixel representation value_
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
    return pixel_array

def resize_img(img_paths: list, target_size: tuple=(128, 128)) -> np.ndarray:
    
    """_Resize and fix pixel array_

        Returns
        -------
        _np.ndarray_
            _Returns fixed and normalized image_
    """
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

def normalize_volume(resized_volume: np.array) -> np.array:

    """_Normalizes the volume of images for going from 0 to 1 values_

    Returns
    -------
    _np.array_
        _returns the same object type but with the normalization applied_
    """
    original_shape = resized_volume.shape
    flattened_image = resized_volume.reshape((-1,))
    scaler = preprocessing.MinMaxScaler()
    normalized_flattened_image = scaler.fit_transform(flattened_image.reshape((-1, 1)))
    normalized_volume_image = normalized_flattened_image.reshape(original_shape)
    return normalized_volume_image

def change_depth_siz(patient_volume: np.ndarray, target_depth: int=64) -> np.ndarray:

    """_Change the depth of an input volume as a numpy array
        considering SIZ algorithm and a desired target depth_

    Returns
    -------
    _np.ndarray_
        _Volume reduced from the original input volume containing
         the target depth as the total number of slices per volume_
    """
    current_depth = patient_volume.shape[0]
    depth = current_depth / target_depth
    depth_factor = 1 / depth
    img_new = zoom(patient_volume, (depth_factor, 1, 1, 1), mode='nearest')
    return img_new

def generate_patient_processed_data(list_img_paths: list, target_size: tuple=(128,128)):

    height = target_size[0]
    width = target_size[1]
    depth = len(list_img_paths)

    volume_array = np.zeros((height, width, depth), dtype=np.float64)

    print("Initializing data preprocessing with the following dimensions-> Volumes:{}".format(volume_array.shape))

    resized_images = resize_img(list_img_paths, target_size=target_size)
    normalized_siz_volume = normalize_volume(resized_images)

    volume_array = normalized_siz_volume
    volume_array = volume_array.transpose(2, 0, 1)
    return np.expand_dims(volume_array, axis=-1)

def __reduce_data_with_prediction__(model, x_data: np.ndarray) -> np.ndarray:

    """_This function takes the model to use for prediction
        and a set of data from which to predict the segmentation
        the model trained in this case was a U-Net and it was 
        trained for segmentation of 6 different classes
        
        The model prediction is then used to delimit an
        upper and lower limit as indexes from the original x_data_

    Returns
    -------
    _np.array_
        _Returns the exact same type of input x_data
         but reduced considering the upper and lower limit_
    """

    #model = load_model(model_path, compile=False)
    
    y_pred = model.predict(x_data)
    y_pred_argmax=np.argmax(y_pred, axis=3)

    upper_limit_class = 1
    lower_limit_class = 5
    
    start_idx = None
    end_idx = None

    for i, _slice in enumerate(y_pred_argmax):
        class_mask = _slice == upper_limit_class
        masked_trues = class_mask * _slice
        summed_pixels = np.sum(masked_trues)
        class_mask_ = _slice == lower_limit_class
        masked_trues_ = class_mask_ * _slice
        summed_pixels_ = np.sum(masked_trues_)

    
        if summed_pixels > 100:
            if start_idx is None:
                start_idx = i

        if summed_pixels_ > 100:
            end_idx = i

    if (start_idx != None) & (end_idx != None):
        thresholded_array = x_data[start_idx+1: end_idx, :, :]

        print("Shape of original array:", x_data.shape)
        print("Shape of thresholded array:", thresholded_array.shape)
        return thresholded_array
    else:
        print("Classes 1 and 5 not found in the array.")

def string_to_list(string_repr):
    return eval(string_repr)

if __name__ == "__main__":
    data = pd.read_csv("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/train_data_lstm.csv")
    data['Patient_paths'] = data['Patient_paths'].apply(string_to_list)
    model = load_model("Unet_Fine_Tune_128_plus_CKP.h5", compile=False)
    for i in range(4):
         print(f'Generating data for patient -> {str(data["Patient_id"][i])} \n')
         patient_data_volumes = generate_patient_processed_data(data["Patient_paths"][i], target_size=(128,128))
         print("Initializing prediction and data reduction \n")
         patient_data_volumes_reduced = __reduce_data_with_prediction__(model, patient_data_volumes)
         print("Changing volume depth with zoom... \n")
         patient_data_volumes_zoom = change_depth_siz(patient_data_volumes_reduced, target_depth=64)
         
         with open(f'D:/Downloads/rsna-2023-abdominal-trauma-detection/volume_data_for_LSTM/{str(data["Patient_id"][i])}_{str(data["Series_id"][i])}.npy', 'wb') as f:
             np.save(f, patient_data_volumes_zoom)
             print(f'Process finished for patient -> {str(data["Patient_id"][i])}', f"Final shape saved: {patient_data_volumes_zoom.shape}")