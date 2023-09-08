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

tf.config.run_functions_eagerly(True)

# ---------------------- Functions for generated cleaned dataframe ----------------------

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