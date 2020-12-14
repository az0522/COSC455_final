"""
This file loads the .npz file and evaluate the cnn model.
!!! The evaluation takes hours to finish !!!
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def show_sample_image(img_folder):
	for i in range(9):
		# define subplot
		plt.subplot(330 + 1 + i)
		# define filename
		filename = img_folder + 'train_' + str(i) + '.jpg'
		# load image pixels
		image = imread(filename)
		# plot raw pixel data
		plt.imshow(image)
	# show the figure
	plt.show()


def load_dataset_test(npz_filename):
	data = np.load(npz_filename)
	X, y = data['arr_0'], data['arr_1']
	trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=12)
	return trainX, testX, trainY, testY


def fbeta(y_true, y_pred, beta=2):
	# clip predictions
	y_pred = backend.clip(y_pred, 0, 1)
	# calculate elements
	tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
	fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
	fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
	# calculate precision
	p = tp / (tp + fp + backend.epsilon())
	# calculate recall
	r = tp / (tp + fn + backend.epsilon())
	# calculate fbeta, averaged across each class
	bb = beta ** 2
	fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
	return fbeta_score


# define cnn model
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
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])
	return model


def summarize_diagnostics(history):
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	plt.subplot(212)
	plt.title('Fbeta')
	plt.plot(history.history['fbeta'], color='blue', label='train')
	plt.plot(history.history['val_fbeta'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_plot.png')
	plt.close()


def run_test_harness(npz_filename):
	# load dataset
	X_train, X_test, y_train, y_test = load_dataset_test(npz_filename)
	# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0 / 255.0)
	# prepare iterators
	train_it = datagen.flow(X_train, y_train, batch_size=128)
	test_it = datagen.flow(X_test, y_test, batch_size=128)
	# define model
	model = define_model()
	# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it),
						validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
	# evaluate model
	loss, fbeta = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> loss=%.3f, f-2=%.3f' % (loss, fbeta))
	# learning curves
	summarize_diagnostics(history)


# folder = 'original/'
# folder = 'dehazed/'
# show_sample_image(folder)


"""
the 'img_dataset.npz' is around 1.5 GB, it is not included in the submission of the final project
'img_dataset.npz' is available at:
"""
npz_filename = 'img_dataset.npz'
run_test_harness(npz_filename)
