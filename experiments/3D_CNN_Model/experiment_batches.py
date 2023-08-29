import tensorflow as tf 
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import sqlite3
import re
from data_genetator import NumpyImage3DGenerator
from model_cnn import ThreeDCNN

# ------------------------------------------ Utility Functions -------------------------------------------------------------------
def string_to_list(string_repr):
    return eval(string_repr) 

def training_plot(metrics, history):
    f, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric], ls='dashed')
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel(metric)
        ax[idx].plot(history.history['val_' + metric]);
        ax[idx].legend([metric, 'val_' + metric])

# -------------------------------------------- Main run ------------------------------------------------------------------------

if __name__ == "__main__":

    connection = sqlite3.connect("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/training_data.db")
    # ATTENTION ABOUT THE TABLE FROM THE DB YOU CONNECT!!
    sql = pd.read_sql_query("SELECT * FROM base_data", connection)
    data = pd.DataFrame(sql, columns =["Patient_id", "Series_id", "Patient_paths", "Patient_category"])
    data['Patient_paths'] = data['Patient_paths'].apply(string_to_list)

    data_frame = pd.read_csv("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/preprocessing/train_data_map.csv")

    batch_sizes = [2**n for n in range(1,8)]
    batch_sizes.reverse()
    input_shape = (128, 128, 64, 1)

    for i, batch_size in enumerate(batch_sizes):
        model = ThreeDCNN(input_shape).model
        #model.summary()
        data_gen = NumpyImage3DGenerator(data_frame["Patient_id"], data_frame["Series_id"], data_frame["Patient_category"], batch_size=batch_size)

        try:
            print(f"Trying fit model with batch_size = {batch_size}")
            model.fit(data_gen, batch_size=4, epochs=10)
            del model
            del data_gen
            continue
        except tf.errors.ResourceExhaustedError:
            if i < len(batch_sizes) - 1:
                print(f"Resources exhausted with batch_size = {batch_size}, trying now with batch_size = {batch_sizes[i+1]}")
            else:
                print(f"Resources exhausted with batch_size = {batch_size}, no more batch_sizes to try")
                break
            continue
