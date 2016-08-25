"""Deep dive libs"""
from input_data_dive_test import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
from underwater_pathfinder import create_structure

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
import glob


# Set path as folder
# Set overlap always rounded down.

config = configDehazenet()
overlap_size = (7, 7)
""" Configuration, set all the variables , including getting all the files that are going to be evaluated. """


im_names =  glob.glob(config.evaluate_path + "*.jpg")
im_names = im_names + glob.glob(config.evaluate_path+ "*.png")

print config.evaluate_path
print im_names

""" Declare the placeholders variables """

x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")

# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()
last_layer, dropoutDict, feature_maps, scalars, histograms = create_structure(tf, x, config.input_size, config.dropout)
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver(tf.all_variables())


""" Recover the previous state of the models. """

if not os.path.exists(config.models_path):
  os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)

print ckpt

if ckpt.model_checkpoint_path:
  print 'Restoring from ', ckpt.model_checkpoint_path  
  saver.restore(sess,ckpt.model_checkpoint_path)
else:
  ckpt = 0

print im_names

for name in im_names:

  im = Image.open(name).convert('RGB')

  im = np.array(im, dtype=np.float32)

  im=im*(1./255.)
  print im
  visualizer = im
  feedDict=dropoutDict
  """ Open one image and add some padding to it """
  
  height, width = im.shape[0], im.shape[1]
  im = np.lib.pad(im, ((0, (config.input_size[0] - (height%config.input_size[0]))
      %config.input_size[0]),(0,(config.input_size[1]-(width%config.input_size[1]))
      %config.input_size[1]), (0,0)), 'constant', constant_values=(0))
  height, width = im.shape[0], im.shape[1]
  im_output=np.zeros((height-config.input_size[0],width-config.input_size[1]),dtype=np.float32)
  #n_tiles=(height-config.input_size[0])*(width-config.input_size[1])
  n_batch=3072
  tile=np.empty((3072,config.input_size[0],config.input_size[1],3),dtype=np.float32)
  altura=height-config.input_size[0]
  largura=width-config.input_size[1]
  inicio=time.time()
  for k in range(0,altura*largura,n_batch):
    for l in range(k,min((k+n_batch),(altura*largura))):
      i=l/largura
      j=l%largura
      tile[l-k] = im[i:i+config.input_size[0], j:j+config.input_size[1]]
    feedDict[x]=tile
    feed_dict=feedDict
    result=sess.run(last_layer, feed_dict=feedDict)
    for l in range(k,min((k+n_batch),(altura*largura))):
      i=l/largura
      j=l%largura
      im_output[i,j]=result[l-k]
    #print im_output[i,j]
  print time.time()-inicio
  result = Image.fromarray((im_output * 255).astype(np.uint8))
  result.save(config.evaluate_out_path + name[len(config.evaluate_path ):])