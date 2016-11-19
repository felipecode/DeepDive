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

def read_depth_image(image_name):
	image = Image.open(image_name).convert('L')
	image = image.resize((input_size[0],input_size[1]), Image.ANTIALIAS)
	image = np.asarray(image)
	image = image.astype(np.float32)
	image = np.multiply(image, 10.0 / 255.0)
	return image

imgs_path='../datasets/simulator_data/images';
depths_path='../datasets/simulator_data/depths';
leveldb_path='../datasets/simulator_data/levelDB';

input_size = (224,224,3)
output_size = (224,224,3)
depth_input_size = (224,224,1)
depth_output_size = (224,224,1)

if not os.path.exists(leveldb_path):
	os.makedirs(leveldb_path)

db = leveldb.LevelDB(leveldb_path + '/db') #Salva antes do training path

im_names = sorted(glob.glob(imgs_path + "/*.png")) #ordenando, so pra garantir
im_names_depth = sorted(glob.glob(depths_path + "/*.png"))

print len(im_names)
print len(im_names_depth)
assert len(im_names) == len(im_names_depth)

n_total_imgs=len(im_names)
n_train_imgs=int(0.95*n_total_imgs)
n_val_imgs=n_total_imgs-n_train_imgs

#primeiros 95% sao de train
db.Put('num_examples',str(n_train_imgs))
for i in xrange(n_train_imgs):
	image=read_image(im_names[i])
	db.Put(str(i),image.tostring())

	depth_image=read_depth_image(im_names_depth[i])
	db.Put(str(i)+"depth",depth_image.tostring())

	#transmission = read_image_gray_scale(im_names_trans[i])
	#db.Put(str(i)+"trans",str(transmission))



#im_names_val = glob.glob(config.validation_path + "/*.jpg")
#im_names_val_labels = glob.glob(config.validation_path_ground_truth + "/*.jpg")
#im_names_trans = glob.glob(config.validation_transmission_path + "/*.jpg")
#assert len(im_names_val) == len(im_names_val_labels)
#5% finais sao validation
print n_val_imgs
db.Put('num_examples_val',str(n_val_imgs))
for i in xrange(n_train_imgs, n_total_imgs):
	image=read_image(im_names[i])
	db.Put("val"+str(i-n_val_imgs),image.tostring())

	depth_image=read_depth_image(im_names_depth[i])
	db.Put("val"+str(i-n_val_imgs)+"depth",depth_image.tostring())
