import tensorflow as tf 
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.layers import Conv3D, MaxPool3D, Flatten, Reshape, Dense
from keras.layers import Dropout, Input, UpSampling3D
#K.set_image_data_format('channels_first')

def encoder_3d(inputs, num_filters):
	x = Conv3D(filters=num_filters, kernel_size=(3,3,3),
							activation="relu",
							padding='same')(inputs)
	x = MaxPool3D(pool_size=(2,2,2))(x)
	return x

def decoder_3d(inputs, num_filters):
	x = Conv3D(filters=num_filters, kernel_size=(3,3,3),
							activation="relu",
							padding='same')(inputs)
	x = UpSampling3D(size=(2,2,2))(x)
	return x

def build_autoencoder(input_shape):
	input_layer = Input(input_shape)

	# encoder
	x1 = encoder_3d(input_layer, 128)
	x2 = encoder_3d(x1, 64)
	x3 = encoder_3d(x2, 64)
	x4 = encoder_3d(x3, 32)
	x5 = encoder_3d(x4, 32)
	x6 = encoder_3d(x5, 32)

	# flatten
	#flatten_layer = Flatten()(x6)
	#dense_layer = Dense(4 * 4 * 4 * 32)(flatten_layer)
	#reshape_layer = Reshape((4, 4, 4, 32))(dense_layer)

	# decoder
	x7 = decoder_3d(x6, 32)
	x8 = decoder_3d(x7, 32)
	x9 = decoder_3d(x8, 32)
	x10 = decoder_3d(x9, 64)
	x11 = decoder_3d(x10, 64)
	output_layer = decoder_3d(x11, 128)

	model = Model(inputs=input_layer, outputs=output_layer, name="3D autoencoder")

	model.compile(loss='mae',
				 optimizer=Adam(),
				  metrics=['acc'])
	return model

if __name__ == "__main__":
	input_shape = (128, 128, 128, 1)
	model = build_autoencoder(input_shape)
	model.summary()