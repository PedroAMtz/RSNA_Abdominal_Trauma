import tensorflow as tf 
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import sqlite3
import re
from data_genetator import NumpyImage3DGenerator
from model_cnn import ThreeDCNN
import numpy as np

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

# ------------------------------------- Learning rate schedule ----------------------------------

def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

def scheduler(epoch, lr):
    if epoch <= 8:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def create_pattern(data):
    pattern = [1, 1, 0, 0]  # Define the pattern (two ones, followed by two zeros)
    result = []
    
    while len(result) < len(data):
        result.extend(pattern)
    
    return result[:len(data)]

class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))
# -------------------------------------------- Main run ------------------------------------------------------------------------

if __name__ == "__main__":

    connection = sqlite3.connect("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/training_data.db")
    # ATTENTION ABOUT THE TABLE FROM THE DB YOU CONNECT!!
    sql = pd.read_sql_query("SELECT * FROM balanced_data", connection)
    data = pd.DataFrame(sql, columns =["Patient_id", "Series_id", "Patient_paths", "Patient_category"])
    data['Patient_paths'] = data['Patient_paths'].apply(string_to_list)


    np.random.seed(10)

    rnd = np.random.rand(len(data))
    train = data[rnd<0.9]
    test = data[(rnd>=0.9)]
    
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train['pattern'] = create_pattern(train)
    train.sort_values(by=['pattern'], inplace=True)
    train.reset_index(drop=True, inplace=True)

    input_shape = (128, 128, 64, 1)
    model = ThreeDCNN(input_shape).model
    #model.summary()


    #print(len(data), len(train), len(test))
    #print(train.head(30))
    #print(test)
    
    with mlflow.start_run() as run:
        #mlflow.set_experiment("Experiment_1_V1")
        mlflow.tensorflow.autolog()

        run_id = run.info.run_id

        checkpoint_filepath = 'D:/Downloads/rsna-2023-abdominal-trauma-detection/Experiments_ckpt/experiment_{}_checkpoint.ckpt'.format(str(run_id))
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True)
        
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=20)

        callbacks = [model_checkpoint_callback,
                     tf.keras.callbacks.LearningRateScheduler(scheduler),
                     PrintLR()]

        data_gen = NumpyImage3DGenerator(train["Patient_id"], train["Series_id"], train["Patient_category"], batch_size=4)
        data_gen_test = NumpyImage3DGenerator(test["Patient_id"], test["Series_id"],test["Patient_category"], batch_size=4)
        #print(data_gen[0])
        #print(data_gen_test[0].shape)
        input_shape = (128, 128, 64, 1)
        model = ThreeDCNN(input_shape).model
        history = model.fit(data_gen, validation_data=data_gen_test, epochs=100, callbacks=callbacks)
        
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id

        # Need to get validation data
        #training_plot(['loss', 'acc'], history)
        #plt.show()