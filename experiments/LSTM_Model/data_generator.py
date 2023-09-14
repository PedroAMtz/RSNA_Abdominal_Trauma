import tensorflow as tf
import pydicom
import cv2
from sklearn import preprocessing
import math
import numpy as np


class SuperDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set: list, y_set: list, batch_size: int, target_size: tuple) -> None:

        """_Initialize the Data Generator_
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.taget_size = target_size
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def standardize_pixel_array(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    
        """_Correct DICOM pixel_array if PixelRepresentation == 1._

        Returns
        -------
        _np.ndarray_
            _returns the pixel array from the dicom file with the
            fixed pixel representation value_
        """
        pixel_array = dcm.pixel_array
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
        return pixel_array
    
    def resize_img(self, img_paths: list) -> np.ndarray:

        """_Resize and fix pixel array_

        Returns
        -------
        _np.ndarray_
            _Returns the volume of fixed images_
        """

        volume_shape = (self.target_size[0], self.target_size[1], len(img_paths)) 
        volume = np.zeros(volume_shape, dtype=np.float64)
        for i, image_path in enumerate(img_paths): 
            image = pydicom.read_file(image_path)
            image = self.standardize_pixel_array(image)
            image = cv2.resize(image, self.target_size)
            normalized_image = image / 255.0
            volume[:,:,i] = normalized_image
        return volume
    
    def __getitem__(self, index):
        batch_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        resized_shape = (len(batch_x), self.target_size[0], self.target_size[1], 1)
        resized_images = np.zeros(resized_shape, dtype=np.float64)
        for i, list_files in enumerate(batch_x):
            preprocessed_images = self.resize_img(list_files)
            resized_images[i,:,:,:] = preprocessed_images
        return resized_images, np.array(batch_y)