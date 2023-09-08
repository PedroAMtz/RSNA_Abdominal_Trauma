import tensorflow as tf 
from keras import backend as K
from keras import backend as K
from keras.optimizers import SGD
from keras.models import load_model, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Conv2DTranspose, concatenate
from keras.layers import Dropout, Input, BatchNormalization

def double_conv_block(x, num_filters):
	# Conv2D then ReLU activation
	x = Conv2D(num_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
	x = Conv2D(num_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
	return x 

def downsample_block(x, num_filters):
	f = double_conv_block(x, num_filters)
	p = MaxPool2D(2)(f)
	p = Dropout(0.3)(p)
	return f, p

def upsample_block(x, conv_features, num_filters):
	x = Conv2DTranspose(num_filters, 3, 2, padding="same")(x)
	x = concatenate([x, conv_features])
	x = Dropout(0.3)(x)
	x = double_conv_block(x, num_filters)
	return x

def build_unet_model(input_shape):
	units = 128
	inputs = Input(input_shape)

	f1, p1 = downsample_block(inputs, units/4)
	f2, p2 = downsample_block(p1, units/2)
	f3, p3 = downsample_block(p2, units*1)
	f4, p4 = downsample_block(p3, units*2)

	bottleneck = double_conv_block(p4, units*4)

	u6 = upsample_block(bottleneck, f4, units*2)
	u7 = upsample_block(u6, f3, units*1)
	u8 = upsample_block(u7, f2, units/2)
	u9 = upsample_block(u8, f1, units/4)

	outputs = Conv2D(6, 1, padding="same", activation="sigmoid")(u9)

	unet_model = Model(inputs, outputs, name="U-Net")

	return unet_model

if __name__ == "__main__":

	input_shape = (128, 128, 1)
	Unet = build_unet_model(input_shape)
	Unet.summary()
