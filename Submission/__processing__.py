
from scipy.ndimage import zoom
import numpy as np
import cv2
import pandas as pd
from glob import glob
import pydicom
from sklearn import preprocessing
import random
import re

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
    
def normalize_volume(resized_volume: np.ndarray) -> np.ndarray:

    """_Normalizes a 2D input image_

        Returns
        -------
        _np.ndarray_
            _returns normalized image as numpy array_
    """
    original_shape = resized_volume.shape
    flattened_image = resized_volume.reshape((-1,))
    scaler = preprocessing.MinMaxScaler()
    normalized_flattened_image = scaler.fit_transform(flattened_image.reshape((-1, 1)))
    normalized_volume_image = normalized_flattened_image.reshape(original_shape)
    return normalized_volume_image

def transpose_and_expand_data(volume_images: np.ndarray) -> np.ndarray:
  
    """_Transpose and expand volume data into a shape following
        (# of images, width, length, channels)_

    Returns
    -------
    _np.ndarray_
        _returns a numpy array transposed and with expanded dimensions_
    """

    transposed_volume_dcm = np.transpose(volume_images, (2, 0, 1))
    transposed_volume_dcm = np.expand_dims(transposed_volume_dcm, axis=3)

    print(f"Final data shape: {transposed_volume_dcm.shape}")

    return transposed_volume_dcm

def change_depth_siz(patient_volume: np.ndarray, target_depth: int=64) -> np.ndarray:

    """_Change the depth of an input volume as a numpy array
        considering SIZ algorithm and a desired target depth_

    Returns
    -------
    _np.ndarray_
        _Volume reduced from the original input volume containing
         the target depth as the total number of slices per volume_
    """
    current_depth = patient_volume.shape[-1]
    depth = current_depth / target_depth
    depth_factor = 1 / depth
    img_new = zoom(patient_volume, (1, 1, depth_factor), mode='nearest')
    return img_new

def extract_number_from_path(path: str):

    """_Auxiliary function that helps process_train_data() function
        to sort image paths_

    Returns
    -------
    _int_
    """

    match = re.search(r'(\d+)\.dcm$', path)
    if match:
        return int(match.group(1))
    return 0

def process_training_data(data: pd.DataFrame, train_data_cat: pd.DataFrame,
                          path: str, number_idx: int, extended_data: bool=False, extract_paths: bool=False) -> pd.DataFrame:

    """_This function process training data based on two input DataFrames
        in this case because data is fragmented in more than one csv file_

    Returns
    -------
    _pd.DataFrame_
        _Returns a pd.DataFrame, if extended_data equal True
        it will return the DataFrame with paths and labels, different from
        normal functionality that returns a list of paths per patient study,
        if extract_paths equal True then the list of paths will have
        a length of 64 random but ordered slices from the study,
        if both parameters are True then it will return the pd.DataFrame
        with paths and labels but paths are reduced by extract_paths condition_
    """
    # Cut columns from dataframe that are redundant or not useful
    data_to_merge = data[["patient_id", "series_id"]]
    patient_category = train_data_cat[['patient_id','bowel_healthy',
                                        'bowel_injury',
                                        'extravasation_healthy',
                                        'extravasation_injury',
                                        'kidney_healthy',
                                        'kidney_low',
                                        'kidney_high',
                                        'liver_healthy',
                                        'liver_low',
                                        'liver_high',
                                        'spleen_healthy',
                                        'spleen_low',
                                        'spleen_high']]
    
    # merge a single DataFrame from input DataFrames
    merged_df = data_to_merge.merge(patient_category, on='patient_id', how='left')
    # Shuffle merged DataFrame
    shuffled_data = merged_df.sample(frac=1, random_state=42)
    shuffled_indexes = shuffled_data.index[:number_idx]
    selected_rows = shuffled_data.loc[shuffled_indexes]
    data_to_merge_processed = selected_rows.reset_index()
    only_labels = data_to_merge_processed[['bowel_healthy',
                                        'bowel_injury',
                                        'extravasation_healthy',
                                        'extravasation_injury',
                                        'kidney_healthy',
                                        'kidney_low',
                                        'kidney_high',
                                        'liver_healthy',
                                        'liver_low',
                                        'liver_high',
                                        'spleen_healthy',
                                        'spleen_low',
                                        'spleen_high']]
    
    total_paths = []
    patient_ids = []
    series_ids = []
    category = []
    # Iterate merged Dataframe and extract image paths, store them in a list
    for patient_id in range(len(data_to_merge_processed)):
    
        p_id = str(data_to_merge_processed["patient_id"][patient_id]) + "/" + str(data_to_merge_processed["series_id"][patient_id])
        str_imgs_path = path + p_id + '/'
        patient_img_paths = []

        for file in glob(str_imgs_path + '/*'):
            patient_img_paths.append(file)
        
        for index, row in only_labels.iterrows():
        # Convert the row to a list and append it to 'row_lists'
            row_list = list(row)
            category.append(row_list)
        # Sort lists paths
        sorted_file_paths = sorted(patient_img_paths, key=extract_number_from_path)
        if extract_paths != False:    
            random_sample_paths = random.sample(sorted_file_paths, 64)
            total_paths.append(random_sample_paths)
        else:
            total_paths.append(sorted_file_paths)
        
        patient_ids.append(data_to_merge_processed["patient_id"][patient_id])
        series_ids.append(data_to_merge_processed["series_id"][patient_id])
    
    final_data = pd.DataFrame(list(zip(patient_ids, series_ids, total_paths, category)),
               columns =["Patient_id", "Series_id", "Patient_paths", "Patient_category"])
    # Different functionality if trainign LSTM
    if extended_data != False:
        
        def update_labels_list(row):
            """_Auxiliary function to create a list from value in category column
                creates a list with lenght equals to the lenght ot the paths lists
                from each patient_

            Parameters
            ----------
            row : _pd.DataFrame_
                _pass dataframe object_

            Returns
            -------
            _list_
                _returns a list with repeated values considering the label (0 or 1)_
            """
            return [row['Patient_category']] * len(row['Patient_paths'])
        
        final_data["Patient_category"] = final_data.apply(update_labels_list, axis=1)
        concatenated_list_paths = [item for sublist in final_data['Patient_paths'] for item in sublist]
        concatenated_list_label = [item for sublist in final_data['Patient_category'] for item in sublist]
        
        data = {"Paths":concatenated_list_paths, "Labels":concatenated_list_label}
        data_new = pd.DataFrame(data)
        
        return data_new
    else:
        return final_data