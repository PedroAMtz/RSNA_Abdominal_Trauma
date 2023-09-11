import pandas as pd 
import numpy as np
import re
import os
from glob import glob
import sqlite3

def extract_number_from_path(path):
    match = re.search(r'(\d+)\.dcm$', path)
    if match:
        return int(match.group(1))
    return 0

def get_data_for_3d_volumes(data, dcm_path, niftii_path):

    data_to_merge = data[["patient_id", "series_id"]]
    
    total_paths = []
    patient_ids = []
    series_ids = []
    seg_path = []
    
    for patient_id in range(len(data_to_merge)):
    
        p_id = str(data_to_merge["patient_id"][patient_id]) + "/" + str(data_to_merge["series_id"][patient_id])
        str_imgs_path = dcm_path + p_id + '/'
        
        seg_mask_paths = niftii_path + str(data_to_merge["series_id"][patient_id]) + ".nii"
        seg_path.append(seg_mask_paths)
        
        patient_img_paths = []

        for file in glob(str_imgs_path + '/*'):
            patient_img_paths.append(file)
        
        
        sorted_file_paths = sorted(patient_img_paths, key=extract_number_from_path)
        total_paths.append(sorted_file_paths)
        patient_ids.append(data_to_merge["patient_id"][patient_id])
        series_ids.append(data_to_merge["series_id"][patient_id])
    
    final_data = pd.DataFrame(list(zip(patient_ids, series_ids, total_paths, seg_path)),
               columns =["patient_id","series_id", "patient_paths", "patient_segmentation"])
    
    return final_data

if __name__ == "__main__":
    
    connection = sqlite3.connect("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/training_data.db")

    metadata_path = "D:/Downloads/rsna-2023-abdominal-trauma-detection/train_series_meta.csv"
    segmentations_path = "D:/Downloads/rsna-2023-abdominal-trauma-detection/segmentations"
    train_metadata = pd.read_csv(metadata_path)
    segmentations = os.listdir(segmentations_path)
    segmentations = [int(os.path.splitext(segmentation)[0]) for segmentation in segmentations]
  
    series = train_metadata["series_id"].tolist()
  
    matched_series = []
    for segmentation in segmentations:
        if segmentation in series:
            matched_series.append(segmentation)
        else:
            continue
  
    patients_segment = train_metadata[train_metadata["series_id"].isin(matched_series)].reset_index(drop=True)
    dcm_path = "D:/Downloads/rsna-2023-abdominal-trauma-detection/train_images/"
    niftii_path = "D:/Downloads/rsna-2023-abdominal-trauma-detection/segmentations/"
    cleaned_data = get_data_for_3d_volumes(patients_segment, dcm_path, niftii_path)
    cleaned_data["patient_paths"] = cleaned_data["patient_paths"].astype(str)

    cleaned_data.to_sql(name="segmentations_data", con=connection, if_exists="append", index=False)
    print("DB table update finished...")