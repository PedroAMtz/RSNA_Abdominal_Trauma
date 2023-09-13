import tensorflow as tf
from keras.utils import to_categorical
import math
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, X_set: np.array, y_set: np.array, num_classes: int, batch_size: int) -> None:
        """_Initialization of data generator_

        Parameters
        ----------
        X_set : np.array
            _Set of images_
        y_set : _np.array_
            _Set of masks_
        num_classes : _int_
            _Number of classes_
        batch_size : _int_
            _Size of the batch taken from X_set and y_set_
        """
        self.x, self.y = X_set, y_set
        self.classes = num_classes
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def __getitem__(self, index):
        """_Get item method for each epoch in training
            and computes categorical values for
            segmentation maks_

        Parameters
        ----------
        index : _int_
            _Integer utilized to return the batches_

        Returns
        -------
        _np.array_
            _Returns the batch from X_set and categorical btach from y_set_
        """
        batch_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        train_masks_cat = to_categorical(batch_y, num_classes=self.classes)
        batch_y_categorical = train_masks_cat.reshape((batch_y.shape[0], batch_y.shape[1], batch_y.shape[2], self.classes))

        return batch_x, batch_y_categorical