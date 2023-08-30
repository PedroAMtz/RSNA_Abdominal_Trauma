import numpy as np
import pandas as pd
import pydicom
import cv2
from scipy.ndimage import zoom
from sklearn import preprocessing
from glob import glob
import re
import sqlite3
"""
This script objective is to process all the images 
and store them as npy files so these files could be 
ingested by the model when training
"""

def extract_number_from_path(path):
    match = re.search(r'(\d+)\.dcm$', path)
    if match:
        return int(match.group(1))
    return 0

def get_data_for_3d_volumes(data, train_data_cat, path, number_idx):

	data_to_merge = data[["patient_id", "series_id"]]
	patient_category = train_data_cat[["patient_id", "any_injury"]]
    
	merged_df = data_to_merge.merge(patient_category, on='patient_id', how='left')
    
	shuffled_data = merged_df.sample(frac=1, random_state=42)
	shuffled_indexes = shuffled_data.index[:number_idx]
	selected_rows = shuffled_data.loc[shuffled_indexes]
	data_to_merge_processed = selected_rows.reset_index()
    
	total_paths = []
	patient_ids = []
	series_ids = []
	category = []
    
	for patient_id in range(len(data_to_merge_processed)):
    
		p_id = str(data_to_merge_processed["patient_id"][patient_id]) + "/" + str(data_to_merge_processed["series_id"][patient_id])
		str_imgs_path = path + p_id + '/'
		patient_img_paths = []

		for file in glob(str_imgs_path + '/*'):
			patient_img_paths.append(file)
        
        
		sorted_file_paths = sorted(patient_img_paths, key=extract_number_from_path)
		total_paths.append(sorted_file_paths)
		patient_ids.append(data_to_merge_processed["patient_id"][patient_id])
		series_ids.append(data_to_merge_processed["series_id"][patient_id])
		category.append(data_to_merge_processed["any_injury"][patient_id])
    
	final_data = pd.DataFrame(list(zip(patient_ids, series_ids, total_paths, category)),
               columns =["Patient_id","Series_id", "Patient_paths", "Patient_category"])
    
	return final_data

def window_converter(image, window_width=400, window_level=50):      
    img_min = window_level - window_width // 2
    img_max = window_level + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    #image = (image / image.max() * 255).astype(np.float64)
    return window_image

def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
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
    
def change_depth_siz(patient_volume, target_depth=64):
	desired_depth = target_depth
	current_depth = patient_volume.shape[-1]
	depth = current_depth / desired_depth
	depth_factor = 1 / depth
	img_new = zoom(patient_volume, (1, 1, depth_factor), mode='nearest')
	return img_new
    
def normalize_volume(resized_volume):
	original_shape = resized_volume.shape
	flattened_image = resized_volume.reshape((-1,))
	scaler = preprocessing.MinMaxScaler()
	normalized_flattened_image = scaler.fit_transform(flattened_image.reshape((-1, 1)))
	normalized_volume_image = normalized_flattened_image.reshape(original_shape)
	return normalized_volume_image

def generate_patient_processed_data(list_img_paths, list_labels, target_size=(128,128), target_depth=64):

    num_patients = len(list_img_paths)
    height = target_size[0]
    width = target_size[1]
    depth = target_depth

    volume_array = np.zeros((height, width, depth), dtype=np.float64)
    labels_array = np.array(list_labels, dtype=np.float64)

    print("Initializing data preprocessing with the following dimensions-> Volumes:{} Labels:{}".format(volume_array.shape, labels_array.shape))

    resized_images = resize_img(list_img_paths, target_size=target_size)
    siz_volume = change_depth_siz(resized_images)
    normalized_siz_volume = normalize_volume(siz_volume)

    volume_array = normalized_siz_volume

    return volume_array, labels_array

def string_to_list(string_repr):
    return eval(string_repr)


if __name__ == "__main__":
	connection = sqlite3.connect("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/training_data.db")
    # ATTENTION ABOUT THE TABLE FROM THE DB YOU CONNECT!!
	sql = pd.read_sql_query("SELECT * FROM base_data", connection)
	data = pd.DataFrame(sql, columns =["Patient_id", "Series_id", "Patient_paths", "Patient_category"])
	data['Patient_paths'] = data['Patient_paths'].apply(string_to_list)

	#cleaned_df = get_data_for_3d_volumes(meta_data, train_data, path=path, number_idx=num_idx)
	#df_to_sql = cleaned_df.copy() 
	#df_to_sql["Patient_paths"] = df_to_sql["Patient_paths"].astype(str)
	#df_to_sql.to_sql(name=f"training_data_{str(num_idx)}", con=conn, if_exists="replace", index=False)

	for i in range(len(data)):
		patient_data_volumes, _ = generate_patient_processed_data(data["Patient_paths"][i],data["Patient_category"][i], target_size=(128,128),target_depth=64)

		with open(f'D:/Downloads/rsna-2023-abdominal-trauma-detection/train_data_128_pilot/{str(data["Patient_id"][i])}_{str(data["Series_id"][i])}.npy', 'wb') as f:
			np.save(f, patient_data_volumes)