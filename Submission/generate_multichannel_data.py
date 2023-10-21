import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from __processing__ import *

def normalize_image(image):
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

def image_window_converter(image):
    windows=[{"window_width": 458, "window_level": 73}, #V_abdomen arterial
           {"window_width": 596, "window_level": 130},  #V_abdomen venoso
            {"window_width": 187, "window_level": 85},  #V_higado arterial
            {"window_width": 230, "window_level": 128}, #V_higado venoso
            {"window_width": 236, "window_level": 118}, #V_bazo venoso
            {"window_width": 309, "window_level": 109}, #V_riñones arterial
            {"window_width": 485, "window_level": 182}, #V_riñones venoso
            {"window_width": 411, "window_level": 95},  #V_pancreas arterial
            {"window_width": 495, "window_level": 202}, #V_pancreas venoso
            {"window_width": 3122, "window_level": 105}, #V_estomago arterial
            {"window_width": 2835, "window_level": 458}] #fx cadera venoso
    imagenes_procesadas=[]
    
    for i, window in enumerate (windows):
        window_width = window["window_width"]
        window_level = window["window_level"]
        
        imagen_procesada = window_converter(image, window_width, window_level)
        img_norm= normalize_image(imagen_procesada)
        imagenes_procesadas.append(img_norm)
        
        window_image = np.stack(imagenes_procesadas, axis=-1)
        
        return window_image

def resize_img(image_path: str, target_size=(128, 128)) -> np.ndarray:
    image = pydicom.read_file(image_path)
    image = standardize_pixel_array(image)
    hu_image = transform_to_hu(image_path, image)
    window_image = image_window_converter(hu_image)
    final_image = cv2.resize(window_image, target_size)
    return final_image

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

def process_training_data(data: pd.DataFrame, train_data_cat: pd.DataFrame, path: str, number_idx: int, extended_data: bool=False, extract_paths: bool=False) -> pd.DataFrame:

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

def string_to_list(string_repr):
    return eval(string_repr)

if __name__ == "__main__":
    data = pd.read_csv("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/train_data_lstm.csv")

    data['Patient_paths'] = data['Patient_paths'].apply(string_to_list)
    data['Patient_category'] = data['Patient_category'].apply(string_to_list)

    for i in range(len(data)):

        patient_data = resize_img(data["Patient_paths"][i])
        labels = np.array(data["Patient_category"][i], dtype=np.float32)

        with open(f'D:/Downloads/rsna-2023-abdominal-trauma-detection/multichannel_data/{str(data["Patient_id"][i])}_{str(data["Series_id"][i])}.npy', 'wb') as f:
            np.save(f, patient_data)
            np.save(f, labels)
            print(f'Process finished for patient -> {str(data["Patient_id"][i])}', f"Final shape saved: {patient_data.shape} and labels {labels.shape}")