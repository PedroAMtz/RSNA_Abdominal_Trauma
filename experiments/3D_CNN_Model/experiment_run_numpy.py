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
    sql = pd.read_sql_query("SELECT * FROM training_data_30", connection)
    data = pd.DataFrame(sql, columns =["Patient_id", "Series_id", "Patient_paths", "Patient_category"])
    data['Patient_paths'] = data['Patient_paths'].apply(string_to_list)
       
    with mlflow.start_run() as run:
        #mlflow.set_experiment("Experiment_1_V1")
        mlflow.tensorflow.autolog()

        run_id = run.info.run_id

        checkpoint_filepath = 'D:/Downloads/rsna-2023-abdominal-trauma-detection/Experiments_ckpt/experiment_{}_checkpoint.ckpt'.format(str(run_id))
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True)
        
        data_gen = NumpyImage3DGenerator(data["Patient_id"], data["Series_id"], data["Patient_category"], batch_size=4)
        input_shape = (128, 128, 64, 1)
        model = ThreeDCNN(input_shape).model
        history = model.fit(data_gen, batch_size=4, epochs=50, callbacks=[model_checkpoint_callback])
        
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id

        # Need to get validation data
        #training_plot(['loss', 'acc'], history)
        #plt.show()