import pandas as pd
import cv2
import os
import numpy as np
from skimage import feature

labels_csv = pd.read_csv('train_labels.csv')
train_img_paths = os.listdir('train/')
test_img_paths = os.listdir('test/')

def process_image(img):
	ret_img = cv2.resize(img, dsize=(500, 300))
	return ret_img

def load_train_images(img_paths, labels_csv):
	images = []
	labels = []
	for i in img_paths:
		row = int(i.split('.')[0])
		plant_class = labels_csv.iloc[row-1]['invasive']

		img = cv2.imread('train/'+i, cv2.IMREAD_GRAYSCALE)
		img = process_image(img)

		images.append(img)
		labels.append(plant_class)
			
	return images, labels

def load_test_images(img_paths):
	images = []
	for i in img_paths:
		img = cv2.imread('test/'+i, cv2.IMREAD_GRAYSCALE)
		img = process_image(img)

		images.append(img)

	return images

def generate_lbp(images, num_points, radius):
	data = []
	for i, img in enumerate(images):
		lbp = feature.local_binary_pattern(img, num_points, radius, method='uniform')
		(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points+3), range=(0, num_points+2))

		hist = hist.astype('float')
		hist /= hist.sum()

		if i % 100:
			print(i, end='\r')
		# cv2.imshow('original image', img)
		# cv2.imshow('lbp image', lbp)
		# cv2.waitKey(1)

		data.append(hist)

	return data

''' generating training data '''
print('loading train images and labels...')
train_images, labels_list = load_train_images(train_img_paths, labels_csv)
print('generating train data...')
train_data = generate_lbp(train_images, 24, 8)
print('loading test images...')
test_images = load_test_images(test_img_paths)
print(test_images[0].shape)
print('generating test data...')
test_data = generate_lbp(test_images, 24, 8)

''' creating training dataframe '''
try:
	train_data = pd.DataFrame(train_data)
	train_data = pd.concat([train_data, pd.DataFrame(labels_list)], axis=1)
	train_data.to_csv('train_data.csv', index=False)

except Exception as e:
	print('Train data could not generated.')

''' creating testing dataframe '''
try:
	test_data = pd.DataFrame(test_data)
	test_data.to_csv('test_data.csv', index=False)
except Exception as e:
	print('Test data could not be generated')
