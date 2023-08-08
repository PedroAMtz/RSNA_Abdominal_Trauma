import tensorflow as tf 
import pandas as pd
import pydicom

"""
This script objective is to process all the images 
and store them as TFrecords so these files could be 
ingested by the model when training
"""

data_path = " "

#data = pd.read_csv(data_path, names=["Filepath", "Patient_id"])

# Use custom function to read dicom file
# then try tf.np_function to see if dataset created
# succesfully)?

def read_dicom(filename):
	file_img = pydicom.read_file(filename)
	image = file_image.pixel_array
	return image


def read_and_decode(filename):
    img = tf.io.read_file(filename)
    img = tf.np_function(read_dicom, [img], tf.float32)
    #img = tf.image.decode_jpeg(img, channels=1)
    #img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def decode_csv(csv_row):
    record_defaults = ["filepaths", "labels"]
    filename, label_string = tf.io.decode_csv(csv_row, record_defaults)
    img = read_and_decode(filename)
    label = tf.argmax(tf.math.equal(["0","1"], label_string))
    return img, label

if __name__ == "__main__":
	print("Let´s try building the tf dataset:)")