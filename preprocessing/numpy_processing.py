import numpy as np
import pandas as pd
import pydicom

"""
This script objective is to process all the images 
and store them as npy files so these files could be 
ingested by the model when training
"""

data_path = " "

data = pd.read_csv(data_path, names=["Filepath", "Patient_id"])

images = []
labels = []


for i in range(len(train_data)):
	file_image = pydicom.read_file(data["Filepath"][i])
	image = file_image.pixel_array
	label = data["Patient_id"][i] # -> The patient_id should map the actual label
    
	images.append(image)
	labels.append(label)

with open('raw_images.npy', 'wb') as f:
	np.save(f, images)
	np.save(f, labels)

if __name__ == "__main__":
	print("Testing")