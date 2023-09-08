import mlflow
from unet import build_unet_model

if __name__	== "__main__":
	input_shape = (128, 128, 1)
	Unet = build_unet_model(input_shape)
	Unet.summary()