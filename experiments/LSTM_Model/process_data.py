import pandas as pd
from glob import glob
import re


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

def process_train_data(data: pd.DataFrame, train_data_cat: pd.DataFrame, path: str, number_idx: int, LSTM_train=False) -> pd.DataFrame:

    """_This function process training data based on two input DataFrames
        in this case because data is fragmented in more than one csv file_

    Returns
    -------
    _pd.DataFrame_
        _Returns a pd.DataFrame, if LSTM_train equal True
        it will return the DataFrame with paths and labels_
    """
    # Cut columns from dataframe that are redundant or not useful
    data_to_merge = data[["patient_id", "series_id"]]
    patient_category = train_data_cat[["patient_id", "any_injury"]]
    # merge a single DataFrame from input DataFrames
    merged_df = data_to_merge.merge(patient_category, on='patient_id', how='left')
    # Shuffle merged DataFrame
    shuffled_data = merged_df.sample(frac=1, random_state=42)
    shuffled_indexes = shuffled_data.index[:number_idx]
    selected_rows = shuffled_data.loc[shuffled_indexes]
    data_to_merge_processed = selected_rows.reset_index()
    
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
        
        # Sort lists paths
        sorted_file_paths = sorted(patient_img_paths, key=extract_number_from_path)
        total_paths.append(sorted_file_paths)
        patient_ids.append(data_to_merge_processed["patient_id"][patient_id])
        series_ids.append(data_to_merge_processed["series_id"][patient_id])
        category.append(data_to_merge_processed["any_injury"][patient_id])
    # Create final DataFrame
    final_data = pd.DataFrame(list(zip(patient_ids, series_ids, total_paths, category)),
               columns =["Patient_id","Series_id", "Patient_paths", "Patient_category"])
    # Different functionality if trainign LSTM
    if LSTM_train == True:
        
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

def process_balanced_data(data: pd.DataFrame)-> pd.DataFrame:
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
        
    data["Patient_category"] = data.apply(update_labels_list, axis=1)
    concatenated_list_paths = [item for sublist in data['Patient_paths'] for item in sublist]
    concatenated_list_label = [item for sublist in data['Patient_category'] for item in sublist]
        
    data = {"Paths":concatenated_list_paths, "Labels":concatenated_list_label}
    data_new = pd.DataFrame(data)
    return data_new

def string_to_list(string_repr):
    return eval(string_repr)