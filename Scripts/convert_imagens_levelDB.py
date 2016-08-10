"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
import gzip
import os
import numpy as np
import Image, colorsys
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import glob
from time import time
from config import *
import leveldb
#import matplotlib.pyplot as plt

def read_image(image_name):
	image = Image.open(image_name).convert('RGB')
	image = image.resize((input_size[0], input_size[1]), Image.ANTIALIAS)
	image = np.asarray(image)
	image = image.astype(np.float32)
	image = np.multiply(image, 1.0 / 255.0)
	return image

def read_image_gray_scale(image_name):
	image = Image.open(image_name).convert('L')
	image = image.resize((input_size[0],input_size[1]), Image.ANTIALIAS)
	image = np.asarray(image)
	image = image.astype(np.float32)
	image = np.multiply(image, 1.0 / 255.0)
	return image

config = configMain()
db = leveldb.LevelDB(config.leveldb_path + 'db') #Salva antes do training path
input_size = config.input_size
output_size = config.output_size

im_names = glob.glob(config.training_path + "/*.jpg")
im_names_labels = glob.glob(config.training_path_ground_truth + "/*.jpg")
im_names_trans = glob.glob(config.training_transmission_path + "/*.jpg")
assert len(im_names) == len(im_names_labels)
db.Put('num_examples',str(len(im_names)))
for i in range(len(im_names)):
	image=read_image(im_names[i])
	db.Put(str(i),image.tostring())
	if len(output_size) == 3:
		image=read_image(im_names_labels[i])
	else:
		image=read_image_gray_scale(im_names_labels[i])
	db.Put(str(i)+"label",image.tostring())
	transmission = np.mean(read_image(im_names_trans[i]))
	db.Put(str(i)+"trans",str(transmission))
	print transmission


im_names_val = glob.glob(config.validation_path + "/*.jpg")
im_names_val_labels = glob.glob(config.validation_path_ground_truth + "/*.jpg")
#im_names_trans = glob.glob(config.validation_transmission_path + "/*.jpg")
assert len(im_names_val) == len(im_names_val_labels)
db.Put('num_examples_val',str(len(im_names_val)))
for i in range(len(im_names_val)):
	image=read_image(im_names_val[i])
	db.Put("val"+str(i),image.tostring())
	if len(output_size) == 3:
		image=read_image(im_names_val_labels[i])
	else:
		image=read_image_gray_scale(im_names_val_labels[i])
	db.Put("val"+str(i)+"label",image.tostring())
	transmission = np.mean(read_image(im_names_trans[i]))
	db.Put("val"+str(i)+"trans",str(transmission))
	print transmission
