import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from PIL import Image

# Resizing images 
def resize_img(img_paths, size=(128, 128)):

	preprocessed_images = []

	for image_path in img_paths:
		image = Image.open(image_path)
		image = image.resize(size)
		image_array = np.array(image)
		preprocessed_images.append(image_array)

# Create an empty volume array
	volume_shape = (size[0], size[1], len(preprocessed_images),  1) 
	volume = np.zeros(volume_shape, dtype=np.uint8)

# Populate the volume with images
	for i, image_array in enumerate(preprocessed_images):
		volume[i] = image_array

	return volume

# Implementation of SIZ algorithm
def change_depth_siz(patient_volume):
	desired_depth = 64
	current_depth = img.shape[-1]
	depth = current_depth / desired_depth
	depth_factor = 1 / depth
	img_new = zoom(img, (1, 1, depth_factor), mode='nearest')
	return img_new
