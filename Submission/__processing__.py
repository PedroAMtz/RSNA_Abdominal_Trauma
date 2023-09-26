
from scipy.ndimage import zoom
import numpy as np
import pandas as pd
from glob import glob
import random
import re

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