"""
This file loads the prediction model (.h5 file), candidate image, label_file(train_v2.csv)
and uses the prediction model to label the candidate image.
"""
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from pandas import read_csv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_tag_mapping(mapping_csv):
	labels = set()
	for i in range(0, len(mapping_csv)):
		tags = mapping_csv['tags'][i].split(' ')
		labels.update(tags)
	labels = list(labels)
	labels.sort()

	labels_map = {labels[i]: i for i in range(len(labels))}
	inv_labels_map = {i: labels[i] for i in range(len(labels))}

	return labels_map, inv_labels_map


def prediction_to_tags(inv_mapping, prediction):
	# round probabilities to {0, 1}
	values = prediction.round()
	# collect all predicted tags
	tags = [inv_mapping[i] for i in range(len(values)) if values[i] == 1.0]
	return tags


# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(128, 128))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 128, 128, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img


# load an image and predict the class
def run_predict(image_name, inv_mapping, prediction_model):
	# load the image
	img = load_image(image_name)
	# load model
	model = load_model(prediction_model)
	# predict the class
	result = model.predict(img)
	print(result[0])
	# map prediction to tags
	tags = prediction_to_tags(inv_mapping, result[0])
	print(tags)


# load the mapping file
# tf.autograph.set_verbosity(0)

tag_filename = 'train_v2.csv'
candidate_image = 'p4.jpg'
prediction_model = 'f.model.h5'

mapping_csv = read_csv(tag_filename)
_, inv_mapping = create_tag_mapping(mapping_csv)
run_predict(candidate_image, inv_mapping, prediction_model)
