"""Deep dive libs"""
from input_data_levelDB import DataSetManager
from config import *
from utils import *
from simulator import *
#from features_optimization import optimize_feature
#from loss_network import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
#from perceptual_loss_network import create_structure

"""Core libs"""
import tensorflow as tf
import numpy as np

"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm


"""Python libs"""
import os
from optparse import OptionParser
from PIL import Image
import subprocess
import time
from ssim_tf import ssim_tf
from scipy import misc
import glob

import json


"""Verifying options integrity"""
config = configMain()

input_size=(224,224)
turbidity_size=(128,128)
batch_size=20
t_batch_size=4
range_max=5.0

if config.restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')
if config.save_features_to_disk not in (True, False):
  raise Exception('Wrong save_features_to_disk option. (True or False)')
if config.save_json_summary not in (True, False):
  raise Exception('Wrong save_json_summary option. (True or False)')
if config.use_tensorboard not in (True, False):
  raise Exception('Wrong use_tensorboard option. (True or False)')

sim_input_path='../datasets/simteste/images'
sim_depth_path='../datasets/simteste/depths'
turbidity_path='../datasets/simteste/TurbidityV3'

inputs = np.empty((batch_size,)+input_size+(3,))
depths = np.empty((batch_size,)+input_size)
turbidities=np.empty((t_batch_size,)+turbidity_size+(3,))

input_names=glob.glob(sim_input_path + "/*.png")
depth_names=glob.glob(sim_depth_path + "/*.png")
t_imgs_names=glob.glob(turbidity_path + "/*.png")

for i in xrange(batch_size):
	in_image = Image.open(input_names[i]).convert('RGB')
	in_image = in_image.resize(input_size, Image.ANTIALIAS)
	in_image = np.asarray(in_image)
	in_image = in_image.astype(np.float32)
	inputs[i] = np.multiply(in_image, 1.0 / 255.0)

	d_image = Image.open(depth_names[i]).convert('L')
	d_image = d_image.resize(input_size, Image.ANTIALIAS)
	d_image = np.asarray(d_image)
	d_image = d_image.astype(np.float32)
	depths[i] = np.multiply(d_image, 10.0 / 255.0)


for i in xrange(t_batch_size):
	t_image = Image.open(t_imgs_names[i]).convert('RGB')
	t_image = t_image.resize(turbidity_size, Image.ANTIALIAS)
	t_image = np.asarray(t_image)
	t_image = t_image.astype(np.float32)
	turbidities[i] = np.multiply(t_image, 1.0 / 255.0)

sess = tf.InteractiveSession()

tf_turbidity=tf.placeholder("float",turbidities.shape, name="turbidity")
properties=acquireProperties(tf_turbidity)

c, binf=sess.run(properties, feed_dict={tf_turbidity: turbidities})

#colocando os vetores no tamanho do batch, nao sei se tem um jeito melhor de fazer isso
c_old=c
c=np.empty((batch_size,c_old.shape[1]))
for i in xrange(batch_size):
	c[i]=c_old[i%len(c_old)]

binf_old=binf
binf=np.empty((batch_size,binf_old.shape[1]))
for i in xrange(batch_size):
	binf[i]=binf_old[i%len(binf_old)]


range_step=range_max/t_batch_size
range_values=np.empty(t_batch_size)
for i in xrange(t_batch_size):
	range_values[i]=(i+1)*range_step

#parte fixa do range
range_array=np.empty(batch_size)
for i in xrange(batch_size):
	range_array[i]=range_values[(i/(batch_size/t_batch_size))%t_batch_size]

tf_images=tf.placeholder("float",inputs.shape, name="images")
tf_depths=tf.placeholder("float",depths.shape, name="depths")
tf_range=tf.placeholder("float",range_array.shape, name="ranges")
tf_c=tf.placeholder("float",c.shape, name="c")
tf_binf=tf.placeholder("float",binf.shape, name="binf")

feedDict={tf_images: inputs, tf_depths: depths, tf_range: range_array, tf_c: c, tf_binf: binf}

turbid_images=applyTurbidity(tf_images, tf_depths, tf_c, tf_binf, tf_range)

start_time =time.time()
result=sess.run(turbid_images, feed_dict=feedDict)
duration=time.time()-start_time

print duration

for i in xrange(batch_size):
	img=result[i]
	img=(img-img.min())
	img*=(255/img.max())
	img=img.astype(np.uint8)
	img = Image.fromarray(img)
	img.save("resultado"+str(i)+".png")

#print np.amax(result[2,:,:,0]), np.amin(result[2,:,:,0])
#print np.amax(result[2,:,:,1]), np.amin(result[2,:,:,1])
#print np.amax(result[2,:,:,2]), np.amin(result[2,:,:,2])
