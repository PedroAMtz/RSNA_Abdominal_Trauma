import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Input, Flatten
from tensorflow.keras.models import Model

def conv_block(input, num_filters):

	x = Conv2D(num_filters, 3, padding="same")(input)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	x = Conv2D(num_filters, 3, padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	return x 

def encoder_block(input, num_filters):

	x = conv_block(input, num_filters)
	p = MaxPool2D((2, 2))(x)
	return x, p

def dense_block(input_shape, num_classes):

	inputs = Input(shape=input_shape)
	
	x = Flatten()(inputs)
	dense_layer_1 = Dense(units=512, activation='relu')(x)
	dense_layer_1 = Dropout(0.4)(dense_layer_1)

	#dense_layer_2 = Dense(units=256, activation='relu')(dense_layer_1)
	#dense_layer_2 = Dropout(0.4)(dense_layer_2)
	output_layer = Dense(units=num_classes, activation='softmax')(dense_layer_1)

	return output_layer

def build_unet_encoder_model(input_shape):

	units = 128
	inputs = Input(input_shape)

	#ENCODER
	s1, p1 = encoder_block(inputs, units/2)
	s2, p2 = encoder_block(p1, units)
	s3, p3 = encoder_block(p2, units*2)
	s4, p4 = encoder_block(p3, units*4)

	b1 = conv_block(p4, units*8)

	encoder_output = b1

	return Model(inputs, encoder_output)



if __name__ == "__main__":

	input_shape = (128, 128, 3)
	num_classes = 2

	encoder = build_unet_encoder_model(input_shape)

	encoder.summary()
