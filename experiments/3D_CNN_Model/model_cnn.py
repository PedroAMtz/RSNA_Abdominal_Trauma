import tensorflow as tf 
from keras import backend as K
from keras import backend as K
from keras.optimizers import SGD
from keras.models import load_model, Model
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization



class ThreeDCNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def convolutional_block_3d(self, inputs, num_filters):
        x = Conv3D(filters=num_filters, kernel_size=(3, 3, 3), activation="relu")(inputs)
        x = MaxPool3D(pool_size=(2, 2, 2))(x)
        x = BatchNormalization()(x)
        return x

    def dense_block(self, flatten_layer):

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

        dense_layer_1 = Dense(units=512, kernel_initializer=initializer, activation='relu')(flatten_layer)
        dense_layer_1 = Dropout(0.4)(dense_layer_1)
        dense_layer_2 = Dense(units=256, activation='relu')(dense_layer_1)
        dense_layer_2 = Dropout(0.4)(dense_layer_2)
        output_layer = Dense(units=13, activation='softmax')(dense_layer_2)
        return output_layer
    
    def compile_model(self, model):
        #optimizer = SGD(learning_rate=1e-06, momentum=0.99, decay=0.0, nesterov=False)
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=['acc', tf.keras.metrics.AUC()])
    
    def summary(self):
        self.model.summary()

    def build_model(self):
        input_layer = Input(self.input_shape)
        x1 = self.convolutional_block_3d(input_layer, 64)
        x2 = self.convolutional_block_3d(x1, 64)
        x3 = self.convolutional_block_3d(x2, 128)
        x4 = self.convolutional_block_3d(x3, 256)
        flatten_layer = Flatten()(x4)
        output = self.dense_block(flatten_layer)
        model = Model(inputs=input_layer, outputs=output)
        self.compile_model(model)
        return model

if __name__ == "__main__":
    input_shape = (128, 128, 64, 1)
    model = ThreeDCNN(input_shape).model
    model.summary()