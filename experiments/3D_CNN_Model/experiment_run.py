import tensorflow as tf 
from keras import backend as K
from glob import glob
import pydicom
from keras import backend as K
from keras.optimizers import SGD
from keras.models import load_model, Model
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
import cv2
from scipy.ndimage import zoom
from sklearn import preprocessing
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import re

# ------------------------------------------ Utility Functions -------------------------------------------------------------------

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

def training_plot(metrics, history):
    f, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric], ls='dashed')
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel(metric)
        ax[idx].plot(history.history['val_' + metric]);
        ax[idx].legend([metric, 'val_' + metric])

# --------------------------------- Data Generator -----------------------------------------------------------------------------

class Image3DGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, target_depth=64, target_size=(128,128)):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.target_size = target_size
        self.target_depth = target_depth

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def standardize_pixel_array(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
        # Correct DICOM pixel_array if PixelRepresentation == 1.
        pixel_array = dcm.pixel_array
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
        return pixel_array

    
    def resize_img(self, img_paths):
        preprocessed_images = []
        for image_path in img_paths: 
            image = pydicom.read_file(image_path)
            image = self.standardize_pixel_array(image)
            image = cv2.resize(image, self.target_size)
            image_array = np.array(image)
            preprocessed_images.append(image_array)

    # Create an empty volume array
        volume_shape = (self.target_size[0], self.target_size[1], len(preprocessed_images)) 
        volume = np.zeros(volume_shape, dtype=np.float64)
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
    
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        resized_images = []
        for list_files in batch_x:
            preprocessed_images = self.resize_img(list_files)
            resized_images_siz = self.change_depth_siz(preprocessed_images)
            normalized_volume = self.normalize_volume(resized_images_siz)
            resized_images.append(normalized_volume)

        resized_images = np.array(resized_images, dtype=np.float64)
        return resized_images, np.array(batch_y, dtype=np.float64)
    
# ---------------------------------- 3D CNN MODEL ---------------------------------------------------------------------------------
def convolutional_block_3d(inputs, num_filters):

    x = Conv3D(filters=num_filters, kernel_size=(3,3,3),
    activation="relu")(inputs)
    x = MaxPool3D(pool_size=(2,2,2))(x)
    x = BatchNormalization()(x)

    return x

# function to define the dense block of the network, composed by:
# 2 dense layer with 2 dropout layes in between and one output layer for clasification
def dense_block(flatten_layer):
    dense_layer_1 = Dense(units=512, activation='relu')(flatten_layer)
    dense_layer_1 = Dropout(0.4)(dense_layer_1)

    #dense_layer_2 = Dense(units=256, activation='relu')(dense_layer_1)
    #dense_layer_2 = Dropout(0.4)(dense_layer_2)
    output_layer = Dense(units=2, activation='softmax')(dense_layer_1)

    return output_layer

# Main function to build the 3D Conv Network
def build_3d_network(input_shape):

    input_layer = Input(input_shape)

    x1 = convolutional_block_3d(input_layer, 64)
    x2 = convolutional_block_3d(x1, 64)
    x3 = convolutional_block_3d(x2, 128)
    x4 = convolutional_block_3d(x3, 256)

    flatten_layer = Flatten()(x4)

    output = dense_block(flatten_layer)
    model = Model(inputs=input_layer, outputs=output)

    model.compile(loss='mae',optimizer=SGD(learning_rate=1e-06, momentum=0.99, decay=0.0, nesterov=False), metrics=['acc'])
    
    return model

# -------------------------------------------- Main run ------------------------------------------------------------------------

if __name__ == "__main__":
       
    with mlflow.start_run() as run:
        mlflow.tensorflow.autolog()

        run_id = run.info.run_id
        
        checkpoint_filepath = 'D:/Downloads/rsna-2023-abdominal-trauma-detection/Experiments_ckpt/experiment_{}_checkpoint.ckpt'.format(str(run_id))
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_accuracy',
                                                                    mode='max',
                                                                    save_best_only=True)

        train_data = pd.read_csv(f"D:/Downloads/rsna-2023-abdominal-trauma-detection/train.csv")
        cat_data = pd.read_csv(f"D:/Downloads/rsna-2023-abdominal-trauma-detection/train_series_meta.csv")
        path = 'D:/Downloads/rsna-2023-abdominal-trauma-detection/train_images/'
        paths = get_data_for_3d_volumes(train_data, cat_data, path=path, number_idx=1600)
    
        train_data_gen = Image3DGenerator(paths["Patient_paths"][:1500], paths["Patient_category"][:1500], batch_size=4)
        valid_data_gen = Image3DGenerator(paths["Patient_paths"][1500:], paths["Patient_category"][1500:], batch_size=4)

        input_shape = (128, 128, 64, 1)
        model = build_3d_network(input_shape)
        history = model.fit(train_data_gen, epochs=100, validation_data=valid_data_gen, callbacks=[model_checkpoint_callback])
        
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id
