import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import re
import os
from glob import glob
from unet import build_unet_model
from sklearn.preprocessing import LabelEncoder
from preprocessing import generate_patient_processed_data
from sklearn.utils import class_weight
import sqlite3

tf.config.run_functions_eagerly(True)

# ------------------------- Custom Loss Function -----------------------------
def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):
        true = K.cast(true, K.floatx())
        pred = K.cast(pred, K.floatx())

        axis = -1 #if channels last 
          #axis=  1 #if channels first


          #argmax returns the index of the element with the greatest value
          #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 
              #if your loss is sparse, use only true as classSelectors

          #considering weights are ordered by class, for each class
          #true(1) if the class index is equal to the weight index 
          #weightsList = tf.cast(weightsList, tf.int64)
        classSelectors = [K.equal(tf.cast(i, tf.int64), tf.cast(classSelectors, tf.int64)) for i in range(len(weightsList))]

          #casting boolean to float for calculations  
          #each tensor in the list contains 1 where ground true class is equal to its index 
          #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

          #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

          #sums all the selections
          #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


          #make sure your originalLossFunc only collapses the class axis
          #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier
        return loss
    return lossFunc

# ------------------------ Plotting utils -----------------------------------

def segmentation_visualization(volume, volume_seg, slice_dcm):
    
    fig = plt.figure(figsize=(14,14), constrained_layout=True)

    ax1 = fig.add_subplot(131)
    ax1.imshow(volume[slice_dcm,:,:], cmap = 'gray')

    ax2 = fig.add_subplot(132)
    ax2.imshow(volume_seg[slice_dcm,:,:], cmap = 'gray')

    ax3 = fig.add_subplot(133)
    ax3.imshow(volume[slice_dcm,:,:]*np.where(volume_seg[slice_dcm,:,:]>0,1,0), cmap = 'gray')
    ax3.set_title('Overlay of Original and Segmented', fontsize=14)
    plt.show()
    
def training_plot(metrics, history):
    f, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric], ls='dashed')
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel(metric)
        ax[idx].plot(history.history['val_' + metric]);
        ax[idx].legend([metric, 'val_' + metric])

if __name__	== "__main__":

  connection = sqlite3.connect("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/training_data.db")
  sql = pd.read_sql_query("SELECT * FROM segmentations_data", connection)
  cleaned_data = pd.DataFrame(sql, columns =["patient_id","series_id", "patient_paths", "patient_segmentation"])
  print(cleaned_data.head())


with mlflow.start_run() as run:
    run_id = run.info.run_id
    mlflow.tensorflow.autolog()
    volume_of_imgs, volume_of_segs = generate_patient_processed_data(cleaned_data["patient_paths"][0], cleaned_data["patient_segmentation"][0])
    labelencoder = LabelEncoder()
    
    n, h, w, _ = volume_of_segs.shape
    train_masks_reshaped = volume_of_segs.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped.ravel())
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    number_classes = len(np.unique(train_masks_reshaped_encoded))
    class_weights = class_weight.compute_class_weight('balanced',
                                                 classes = np.unique(train_masks_reshaped_encoded),
                                                 y = train_masks_reshaped_encoded)
    

    X_train , X_test, y_train, y_test = train_test_split(volume_of_imgs, volume_of_segs, test_size = 0.10, random_state = 0)
    train_masks_cat = to_categorical(y_train, num_classes=number_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], number_classes))
    test_masks_cat = to_categorical(y_test, num_classes=number_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], number_classes))
    

    input_shape = (128, 128, 1)
    Unet = build_unet_model(input_shape)
    

    Unet.compile(optimizer='adam', loss=weightedLoss(tf.keras.losses.categorical_crossentropy, class_weights), metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=number_classes)])
    history = Unet.fit(X_train, y_train_cat, 
                    batch_size = 32, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test_cat), 
                    shuffle=False)
    
    Unet.save(f'Unet_{str(run_id)}.hdf5')
    
    assert mlflow.active_run()
    assert mlflow.active_run().info.run_id == run.info.run_id