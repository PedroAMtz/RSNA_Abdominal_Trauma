import tensorflow as tf
import pydicom
import cv2
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np


class SuperDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set: list, y_set: list, batch_size: int, target_size: tuple, window_width: int, window_level: int) -> None:

        """_Initialize the Data Generator_

        Update -> Now the Window_Level & Window_Width must be passed for datagen initialization
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.target_size = target_size
        self.window_width = window_width
        self.window_level = window_level
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def window_converter(self, image):

        """_Uses the window values in order to create desired contrast to the image_

        Returns
        -------
        _np.ndarray_
            _returns a numpy array with the desired window level applied_
        """

        img_min = self.window_level - self.window_width // 2
        img_max = self.window_level + self.window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        return window_image

    def transform_to_hu(self, medical_image, image):

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
    
    def resize_img(self, image_path: str) -> np.ndarray:

        """_Resize and fix pixel array_

        Returns
        -------
        _np.ndarray_
            _Returns fixed and normalized image_
        """
        image = pydicom.read_file(image_path)
        image = self.standardize_pixel_array(image)
        hu_image = self.transform_to_hu(image_path, image)
        window_image = self.window_converter(hu_image)
        final_image = cv2.resize(window_image, self.target_size)
        return final_image
    
    def normalize_image(self, image):

        """_Normalizes a 2D input image_

        Returns
        -------
        _np.ndarray_
            _returns normalized image as numpy array_
        """

        # Ensure the input image is 2D
        if len(image.shape) != 2:
            raise ValueError("Input must be a 2D image.")
        # Reshape the 2D image into a 1D array
        flattened_image = image.reshape((-1,))
        # Create a MinMaxScaler instance
        scaler = MinMaxScaler()
        # Fit and transform the flattened image
        normalized_flattened_image = scaler.fit_transform(flattened_image.reshape((-1, 1)))
        # Reshape the normalized image back to its original shape
        normalized_image = normalized_flattened_image.reshape(image.shape)
        return normalized_image
    
    def __getitem__(self, index):

        batch_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        resized_shape = (len(batch_x), self.target_size[0], self.target_size[1])
        resized_images = np.zeros(resized_shape, dtype=np.float64)
        for i, file_name in enumerate(batch_x):
            preprocessed_image = self.resize_img(file_name)
            normalized_image = self.normalize_image(preprocessed_image)
            resized_images[i,:,:] = normalized_image
        return np.expand_dims(resized_images, -1), np.array(batch_y, dtype=np.float64)