import numpy as np
import pandas as pd
import pydicom
import cv2
from scipy.ndimage import zoom
from sklearn import preprocessing
from glob import glob

"""
This script objective is to process all the images 
and store them as npy files so these files could be 
ingested by the model when training
"""

def get_data_for_3d_volumes(data, path, number_idx):
    
	data_to_merge = data[["patient_id", "any_injury"]]
	shuffled_data = data_to_merge.sample(frac=1, random_state=42)
	shuffled_indexes = shuffled_data.index[:number_idx]
	selected_rows = shuffled_data.loc[shuffled_indexes]
	data_to_merge_processed = selected_rows.reset_index()
    
	total_paths = []
	patient_ids = []
	category = []
    
	for patient_id in range(len(data_to_merge_processed)):
    
		p_id = str(data_to_merge_processed["patient_id"][patient_id])
		str_imgs_path = path + p_id + '/'
		patient_img_paths = []

		for file in glob(str_imgs_path + '*'):
			for image_path in glob(file + '/*'):
				patient_img_paths.append(image_path)
    
		total_paths.append(patient_img_paths)
		patient_ids.append(data_to_merge_processed["patient_id"][patient_id])
		category.append(data_to_merge_processed["any_injury"][patient_id])
    
	final_data = pd.DataFrame(list(zip(patient_ids, total_paths, category)),
               columns =["Patient_id", "Patient_paths", "Patient_category"])
    
	return final_data

def resize_img(self, img_paths):
	preprocessed_images = []
	for image_path in img_paths: 
		image = pydicom.read_file(image_path)
		image = image.pixel_array
		image = cv2.resize(image, self.target_size)
		image_array = np.array(image)
		preprocessed_images.append(image_array)

    # Create an empty volume array
	volume_shape = (self.target_size[0], self.target_size[1], len(preprocessed_images)) 
	volume = np.zeros(volume_shape, dtype=np.uint16)
    # Populate the volume with images
	for i, image_array in enumerate(preprocessed_images):
		volume[:,:,i] = image_array
	return volume
    
def change_depth_siz(self, patient_volume):
	desired_depth = self.target_depth
		current_depth = patient_volume.shape[-1]
	depth = current_depth / desired_depth
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

def generate_processed_data(list_img_paths, list_labels, target_size=(128,128), target_depth=64):

	num_patients = len(list_img_paths)
	height = target_size[0]
	width = target_size[1]
	depth = target_depth

	volume_array = np.zeros((num_patients, height, width, depth), dtype=np.float64)
	labels_array = np.zeros((num_patients), dtype=np.float64)

	print("Initializing data preprocessing with the following dimensions-> Volumes:{} Labels:{}".format(volume_array.shape, labels_array.shape))

	for i, list_paths in enumerate(list_img_paths):

		resized_images = resize_img(list_paths)
		siz_volume = change_depth_siz(resized_images)
		normalized_siz_volume = normalize_volume(siz_volume)

		volume_array[i] = normalized_siz_volume
		labels_array[i] = list_labels[i]

		if (i + 1) % 10 == 0:
			print(f"Iteration {i + 1}: Data Preprocessing running succesfully...")

	return volume_array, labels_array



if __name__ == "__main__":

	train_data = pd.read_csv(f"D:/Downloads/rsna-2023-abdominal-trauma-detection/train.csv")
	path = 'D:/Downloads/rsna-2023-abdominal-trauma-detection/train_images/'
	cleaned_df = get_data_for_3d_volumes(train_data, path=path, number_idx=2000)

	data_volumes, data_labels = generate_processed_data(cleaned_df["Patient_id"],
														cleaned_df["Patient_category"],
														target_size=(128,128),
														target_depth=64)

	with open(f'3D_data_{str(data_volumes.shape[1])}_{str(data_volumes.shape[2])}_{str(data_volumes.shape[2])}.npy', 'wb') as f:
		np.save(f, data_volumes)
		np.save(f, data_labels)