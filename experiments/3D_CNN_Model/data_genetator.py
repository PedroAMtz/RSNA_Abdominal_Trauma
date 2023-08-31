import tensorflow as tf 
import pydicom
import cv2
from scipy.ndimage import zoom
from sklearn import preprocessing
import math
import numpy as np


class RawImage3DGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, target_depth=64, target_size=(128,128)):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.target_size = target_size
        self.target_depth = target_depth

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def standardize_pixel_array(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    # Correct DICOM pixel_array if PixelRepresentation == 1.
        pixel_array = dcm.pixel_array
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
        return pixel_array

    
    def resize_img(self, img_paths):
        volume_shape = (self.target_size[0], self.target_size[1], len(img_paths)) 
        volume = np.zeros(volume_shape, dtype=np.float64)
        for i, image_path in enumerate(img_paths): 
            image = pydicom.read_file(image_path)
            image = self.standardize_pixel_array(image)
            image = cv2.resize(image, self.target_size)
            volume[:,:,i] = image
        return volume
    
    def change_depth_siz(self, patient_volume):
        current_depth = patient_volume.shape[-1]
        depth = current_depth / self.target_depth
        depth_factor = 1 / depth
        img_new = zoom(patient_volume, (1, 1, depth_factor), mode='nearest')
        return img_new
    
    def normalize_volume(self, resized_volume):
        original_shape = resized_volume.shape
        flattened_image = resized_volume.reshape((-1,))
        scaler = preprocessing.MinMaxScaler()
        normalized_flattened_image = scaler.fit_transform(flattened_image.reshape((-1, 1)))
        normalized_volume_image = normalized_flattened_image.reshape(original_shape)
        return normalized_volume_image
    
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        resized_shape = (len(batch_x), self.target_size[0], self.target_size[1], self.target_depth)
        resized_images = np.zeros(resized_shape, dtype=np.float64)
        for i, list_files in enumerate(batch_x):
            preprocessed_images = self.resize_img(list_files)
            resized_images_siz = self.change_depth_siz(preprocessed_images)
            normalized_volume = self.normalize_volume(resized_images_siz)
            resized_images[i,:,:,:] = normalized_volume
        return resized_images, np.array(batch_y)


class NumpyImage3DGenerator(tf.keras.utils.Sequence):

    def __init__(self, patient_set, series_set, category_set, batch_size):
        self.x, self.y = patient_set, category_set
        self.series = series_set
        self.batch_size = batch_size
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_of_volumes = []
        for patient in range(len(batch_x)):
            try:
                with open(f'D:/Downloads/rsna-2023-abdominal-trauma-detection/train_data_128/{self.x[patient]}_{self.series[patient]}.npy', 'rb') as f:
                    X = np.load(f, allow_pickle=True)
                batch_of_volumes.append(X)
            except:
                continue
                
        return np.array(batch_of_volumes, dtype=np.float64), np.array(batch_y, dtype=np.float64)
    
class NumpyImage3DGeneratorVal():

    def __init__(self, patient_set, series_set, batch_size):
        self.x_v = patient_set
        self.series_v  = series_set
        self.batch_size_v = batch_size
    
    def __len__(self):
        return math.ceil(len(self.x_v) / self.batch_size_v)
    
    def __getitem__(self, idx):
        batch_x_v = self.x_v[idx * self.batch_size_v:(idx + 1) * self.batch_size_v]
        batch_of_volumes_v = []
        for x in range(len(batch_x_v)):
            with open(f'D:/Downloads/rsna-2023-abdominal-trauma-detection/train_data_128/{self.x_v[x]}_{self.series_v[x]}.npy', 'rb') as f:
                X = np.load(f, allow_pickle=True)
            batch_of_volumes_v.append(X)
                
        return np.array(batch_of_volumes_v, dtype=np.float64)