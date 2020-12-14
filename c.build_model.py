"""
This file is used to load the .npz file and use the .npz file to create the prediction model.
The prediction model is saved as 'f.model.h5'
!!! It takes hours to build the prediction model. !!!
"""


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_dataset(npz_filename):
	# load dataset
	data = np.load(npz_filename)
	X, y = data['arr_0'], data['arr_1']
	return X, y


def define_model(in_shape=(128, 128, 3), out_shape=17):
	# load model
	model = VGG16(include_top=False, input_shape=in_shape)
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# allow last vgg block to be trainable
	model.get_layer('block5_conv1').trainable = True
	model.get_layer('block5_conv2').trainable = True
	model.get_layer('block5_conv3').trainable = True
	model.get_layer('block5_pool').trainable = True
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(out_shape, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy')
	return model


def create_prediction_model(npz_filename, model_name):
	# load dataset
	X, y = load_dataset(npz_filename)
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True, horizontal_flip=True, vertical_flip=True, rotation_range=90)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
	train_it = datagen.flow(X, y, batch_size=128)
	# define model
	model = define_model()
	# fit model
	model.fit(train_it, steps_per_epoch=len(train_it), epochs=50, verbose=0)
	# save model
	model.save(model_name)


"""
the 'img_dataset.npz' is around 1.5 GB, it is not included in the submission of the final project
'img_dataset.npz' is available at:

the 'f.model.h5' is included in the submission.
"""
npz_name = 'img_dataset.npz'
model_name = 'f.model.h5'

create_prediction_model(npz_name, model_name)
