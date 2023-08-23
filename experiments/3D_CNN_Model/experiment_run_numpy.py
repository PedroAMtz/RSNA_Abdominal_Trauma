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


# ------------------------------------------ Utility Functions -------------------------------------------------------------------

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

    with open('/kaggle/working/3D_data_128_128_64.npy', 'rb') as f:
        X = np.load(f, allow_pickle=True)
        y = np.load(f, allow_pickle=True)
    print(X.shape, y.shape)
       
    with mlflow.start_run() as run:
        mlflow.set_experiment("Experiment_1_V1")
        mlflow.tensorflow.autolog()

        run_id = run.info.run_id

        checkpoint_filepath = 'Experiments_ckpt/experiment_{}_checkpoint.ckpt'.format(str(run_id))
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_accuracy',
                                                                    mode='max',
                                                                    save_best_only=True)

        input_shape = (128, 128, 64, 1)
        model = build_3d_network(input_shape)
        history = model.fit(X, y, batch_size=4, epochs=1000, validation_data=valid_data_gen, callbacks=[model_checkpoint_callback])
        
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id

        training_plot(['loss', 'acc'], history)