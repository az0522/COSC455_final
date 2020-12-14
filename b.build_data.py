"""
This file is used to import de-hazed images under folder 'train_jpg' folder
and import truth file 'train_v2.csv' to create the single .npz file which
contains the de-hazed images and their corresponding labels.
"""
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from os import listdir
import pandas as pd
import numpy as np


def create_tag_mapping(csv_file):
	mapping_csv = pd.read_csv(csv_file)
	labels = set()
	for i in range(0, len(mapping_csv)):
		tags = mapping_csv['tags'][i].split(' ')
		labels.update(tags)
	labels = list(labels)
	labels.sort()

	labels_map = {labels[i]: i for i in range(len(labels))}
	inv_labels_map = {i: labels[i] for i in range(len(labels))}

	return labels_map, inv_labels_map


def create_file_mapping(csv_file):
	mapping_csv = pd.read_csv(csv_file)
	mapping = dict()
	for i in range(0, len(mapping_csv)):
		name, tags = mapping_csv['image_name'][i], mapping_csv['tags'][i]
		mapping[name] = tags.split(' ')
	return mapping


def hot_encode(tags, mapping):
	encoding = np.zeros(len(mapping), dtype='uint8')
	for tag in tags:
		encoding[mapping[tag]] = 1
	return encoding


def load_data(path, file_mapping, tag_mapping):
	photos, targets = [], []
	for filename in listdir(path):
		photo = load_img(path + filename, target_size=(128, 128))
		photo = img_to_array(photo, dtype='uint8')
		tags = file_mapping[filename[:-4]]
		target = hot_encode(tags, tag_mapping)

		photos.append(photo)
		targets.append(target)

	X = np.asarray(photos, dtype='uint8')
	y = np.asarray(targets, dtype='uint8')

	return X, y


def npz_generator(csv_file, img_folder, npz_name):
	# filename = csv_file
	# mapping_csv = pd.read_csv(filename)
	tag_mapping, _ = create_tag_mapping(csv_file)
	file_mapping = create_file_mapping(csv_file)

	folder = img_folder
	X, y = load_data(folder, file_mapping, tag_mapping)
	print(X.shape, y.shape)
	np.savez_compressed(npz_name, X, y)


csv_file = 'train_v2.csv'

"""
the 'dehazed_all" folder is around 1 GB , it is not included in the submission of the final project.
'dehazed_all' is available at:

the 'img_dataset.npz' is around 1.5 GB, it is not included in the submission of the final project
'img_dataset.npz' is available at:
"""
img_folder = 'dehazed_all/'
npz_name = 'img_dataset'

npz_generator(csv_file, img_folder, npz_name)

# labels_map, inv_labels_map = create_tag_mapping(csv_file)
# for key in labels_map.keys():
# 	print(key)
# print(labels_map)
# print(inv_labels_map)
