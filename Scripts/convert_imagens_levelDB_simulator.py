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
	image = image.resize((config.input_size[0], config.input_size[1]), Image.ANTIALIAS)
	image = np.asarray(image)
	image = image.astype(np.float32)
	image = np.multiply(image, 1.0 / 255.0)
	return image

def read_depth_image(image_name):
	image = Image.open(image_name).convert('I')
	image = image.resize((config.input_size[0],config.input_size[1]), Image.ANTIALIAS)
	image = np.asarray(image,dtype=np.uint16)
	
	image = image.astype(np.float32)
	image = np.multiply(image, 1.0 / 1000.0)
	#print image
	return image

config=configSimConvert()

if not os.path.exists(config.leveldb_path):
	os.makedirs(config.leveldb_path)

db = leveldb.LevelDB(config.leveldb_path + '/db') #Salva antes do training path
print config.imgs_path + "/*.png"
im_names = sorted(glob.glob(config.imgs_path + "/*.png")) + sorted(glob.glob(config.imgs_path2 + "/*.png")) + sorted(glob.glob(config.imgs_path3 + "/*.png"))#ordenando, so pra garantir
im_names_depth = sorted(glob.glob(config.depths_path + "/*.png")) + sorted(glob.glob(config.depths_path2 + "/*.png")) + sorted(glob.glob(config.depths_path3 + "/*.png")) 
print len(glob.glob(config.imgs_path + "/*.png"))
print len(glob.glob(config.imgs_path2 + "/*.png"))
print len(glob.glob(config.imgs_path3 + "/*.png"))
print len(im_names)
print len(im_names_depth)
assert len(im_names) == len(im_names_depth)

n_total_imgs=len(im_names)
n_train_imgs=int(0.90*n_total_imgs)
n_val_imgs=int(0.05*n_total_imgs)
n_test_imgs=n_total_imgs-(n_val_imgs+n_train_imgs)

#primeiros 90% sao de train
db.Put('num_examples',str(n_train_imgs))
for i in xrange(n_train_imgs):
	image=read_image(im_names[i])
	db.Put(str(i),image.tostring())

	depth_image=read_depth_image(im_names_depth[i])
	db.Put(str(i)+"depth",depth_image.tostring())

print n_val_imgs
db.Put('num_examples_val',str(n_val_imgs))
for i in xrange(n_train_imgs, n_total_imgs):
	image=read_image(im_names[i])
	db.Put("val"+str(i-n_train_imgs),image.tostring())

	depth_image=read_depth_image(im_names_depth[i])
	db.Put("val"+str(i-n_train_imgs)+"depth",depth_image.tostring())

print n_test_imgs
db.Put('num_examples_val',str(n_test_imgs))
for i in xrange(n_train_imgs+n_val_imgs, n_total_imgs):
	image=read_image(im_names[i])
	db.Put("val"+str(i-(n_train_imgs+n_val_imgs)),image.tostring())

	depth_image=read_depth_image(im_names_depth[i])
	db.Put("val"+str(i-(n_train_imgs+n_val_imgs))+"depth",depth_image.tostring())
