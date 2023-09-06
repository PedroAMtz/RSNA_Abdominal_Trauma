import streamlit as st
import pandas as pd
import numpy as np
from segmentations_list import segmentations_paths

st.title("RSNA Abdominal Trauma Detection: Segmentations analysis :bar_chart:")

st.write("The purpose of this application is to analyze the segmentation files provided\n" 
		"by the RSNA Abdominal Trauma Competition")

st.subheader("Segmentations Data Analysis :mag_right:")
st.write("This is how the csv file looks like...")

metadata_path = "train_series_meta.csv"

with open(f'25102_50875.npy', 'rb') as f:
	X = np.load(f, allow_pickle=True)

def show_metadata(path):

	train_metadata = pd.read_csv(path)
	return train_metadata.head()

def match_series(path):

	train_metadata = pd.read_csv(path)
	series = train_metadata["series_id"].tolist()

	matched_series = []

	for segmentation in segmentations_paths:
		if segmentation in series:
			matched_series.append(segmentation)
		else:
			continue
	patients_segment = train_metadata[train_metadata["series_id"].isin(matched_series)].reset_index(drop=True)
	return patients_segment

st.dataframe(show_metadata(metadata_path))

st.write(f"The segmentations directory has a total of: **{len(segmentations_paths)}** files")
st.write("### The matched series_id from the segmentations files are:")

st.dataframe(match_series(metadata_path))

st.subheader("Exploring DICOM and NIFTII data")

option = st.selectbox("Which patient yo want to analyze?", 
					match_series(metadata_path)["patient_id"].unique())


instance_number = st.slider("Slice to display..", min_value=0, max_value=X.shape[-1] -1)
st.image(X[:,:,instance_number], width=400)