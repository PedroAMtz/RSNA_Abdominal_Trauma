import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, patient_set: list, series_set: list, batch_size: int):
        self.x= patient_set
        self.series = series_set
        self.batch_size = batch_size
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_of_volumes = []
        batch_of_labels = []
        for patient in range(len(batch_x)):
            try:
                with open(f'D:/Downloads/rsna-2023-abdominal-trauma-detection/Features_for_LSTM/{self.x[patient]}_{self.series[patient]}.npy', 'rb') as f:
                    X = np.load(f, allow_pickle=True)
                    y = np.load(f, allow_pickle=True)
                batch_of_volumes.append(X)
                batch_of_labels.append(y)
            except:
                continue
                
        return np.array(batch_of_volumes, dtype=np.float64), np.array(batch_of_labels, dtype=np.float64)

# Lo de abajo solo para probar que funcione
"""
if __name__ == "__main__":
    data = pd.read_csv("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/train_data_lstm.csv")
    datagen = DataGenerator(data['Patient_id'], data['Series_id'], 64)

    x, y = datagen[0]
    print(x.shape, y.shape)
"""

