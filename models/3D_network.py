import tensorflow as tf 
from keras import backend as K
from keras.optimizers import SGD
from keras.models import load_model, Model
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization

# function to define the convolutional block, composed by:
# 3D convolution, Max pooling 3D, Batch Normalization
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

	dense_layer_2 = Dense(units=256, activation='relu')(dense_layer_1)
	dense_layer_2 = Dropout(0.4)(dense_layer_2)
	output_layer = Dense(units=2, activation='softmax')(dense_layer_2)

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

	model = Model(inputs=input_layer, outputs=output, name="3D CNN")

	model.compile(loss='mae',
				 optimizer=SGD(learning_rate=1e-06, momentum=0.99, decay=0.0, nesterov=False),
				  metrics=['acc'])

	return model

if __name__ == "__main__":

	input_shape = (128, 128, 64, 1)
	model = build_3d_network(input_shape)
	model.summary()