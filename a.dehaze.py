"""
This file loads all images under 'train_all_jpg' folder and de-haze them.
The de-hazed images are stored under 'dehazed_all' folder.

This program implement single image de-hazing using "dark channel prior".
https://github.com/He-Zhang/image_dehaze
"""

from fastai.basics import *
import numpy as np
import math
import cv2


def DarkChannel(im, sz):
	b, g, r = cv2.split(im)
	dc = cv2.min(cv2.min(r, g), b)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
	dark = cv2.erode(dc, kernel)
	return dark


def AtmLight(im, dark):
	[h, w] = im.shape[:2]
	imsz = h * w
	numpx = int(max(math.floor(imsz / 1000), 1))
	darkvec = dark.reshape(imsz, 1)
	imvec = im.reshape(imsz, 3)

	indices = darkvec.argsort()
	indices = indices[imsz - numpx::]

	atmsum = np.zeros([1, 3])
	for ind in range(1, numpx):
		atmsum = atmsum + imvec[indices[ind]]

	A = atmsum / numpx
	return A


def TransmissionEstimate(im, A, sz):
	omega = 0.95
	im3 = np.empty(im.shape, im.dtype)

	for ind in range(0, 3):
		im3[:, :, ind] = im[:, :, ind] / A[0, ind]

	transmission = 1 - omega * DarkChannel(im3, sz)
	return transmission


def Guidedfilter(im, p, r, eps):
	mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
	mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
	mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
	cov_Ip = mean_Ip - mean_I * mean_p

	mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
	var_I = mean_II - mean_I * mean_I

	a = cov_Ip / (var_I + eps)
	b = mean_p - a * mean_I

	mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
	mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

	q = mean_a * im + mean_b
	return q


def TransmissionRefine(im, et):
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	gray = np.float64(gray) / 255
	r = 60
	eps = 0.0001
	t = Guidedfilter(gray, et, r, eps)

	return t


def Recover(im, t, A, tx=0.1):
	res = np.empty(im.shape, im.dtype)
	t = cv2.max(t, tx)

	for ind in range(0, 3):
		res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

	return res


def cv2_to_plt(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def dehaze(image_path, output_path):
	src = cv2.imread(str(image_path))

	I = src.astype('float64') / 255
	dark = DarkChannel(I, 15)
	A = AtmLight(I, dark)
	te = TransmissionEstimate(I, A, 15)
	t = TransmissionRefine(src, te)
	J = Recover(I, t, A, 0.1)

	cv2.imwrite(str(output_path / image_path.name), J * 255)


def batch_dehaze(src_folder, dst_folder):
	images_path = get_image_files(src_folder)

	# de-haze the original images
	for image_path in images_path:
		dehaze(image_path, dst_folder)


"""
the 'train_all_jpg" folder is around 700 MB, it is not included in the submission of the final project.
'train_jpg/' is available at:

the 'dehazed_all" folder is around 1 GB, it is not included in the submission of the final project.
'dehazed_all' is available at:
"""
# src_folder = Path('train_all_jpg')
# dst_folder = Path('dehazed_all')


src_folder = Path('original')
dst_folder = Path('dehazed')
batch_dehaze(src_folder, dst_folder)
