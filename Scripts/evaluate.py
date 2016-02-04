"""Deep dive libs"""
from input_data_dive import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
from dirt_or_rain_structure import create_structure

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

overlap_size = (12, 12)


path = '/home/nautec/DeepDive/Simulator/Dataset1_2/Evaluation/'
out_path ='/home/nautec/DeepDive/Simulator/Dataset1_2/EvaluationResults/'
im_names =  glob.glob(path + "*.jpg")

x = tf.placeholder("float", shape=[None, np.prod(np.array(input_size))], name="input_image")
y_ = tf.placeholder("float", shape=[None, np.prod(np.array(output_size))], name="output_image")

# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()



h_conv3 = create_structure(tf, x,input_size)

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver(tf.all_variables())

if not os.path.exists(models_path):
  os.mkdir(models_path)
ckpt = tf.train.get_checkpoint_state(models_path)
if ckpt and ckpt.model_checkpoint_path and restore:
  print 'Restoring from ', ckpt.model_checkpoint_path  
  saver.restore(sess, models_path + ckpt.model_checkpoint_path)

count =0
for i in im_names:

  im = Image.open(i).convert('RGB')

  im = np.array(im, dtype=np.float32)
  visualizer = im

  im = np.lib.pad(im, ((input_size[0]-overlap_size[0], 0), (input_size[1]-overlap_size[1], 0), (0,0)), mode='constant', constant_values=1)

  original = im
  original = original.astype(np.float32)
  original = np.multiply(original, 1.0 / 255.0)

  height, width = im.shape[0], im.shape[1]

  united_images = np.zeros((im.shape[0]+input_size[0], im.shape[1]+input_size[1], 3), dtype=np.float32)
  out = np.zeros([output_size[0], output_size[1], output_size[2]]).reshape([1, np.prod(np.array(output_size))])

  """Separating the image in chunks"""
  for h in range(0, height, input_size[0]-(overlap_size[0]*2)):
    for w in range(0, width, input_size[1]-(overlap_size[1]*2)):
      h_end = h + input_size[0]
      w_end = w + input_size[1]

      chunk = original[h:h_end, w:w_end]
      if chunk.shape != input_size:
        chunk = np.lib.pad(chunk, ((0, input_size[0]-chunk.shape[0]), (0, input_size[1]-chunk.shape[1]), (0,0)), mode='constant', constant_values=1)

      im = chunk.reshape([1, np.prod(np.array(input_size))])

      result = sess.run(h_conv3, feed_dict={x: im, y_: out})

      united_images[h:h+input_size[0]-(overlap_size[0]*2), w:w+input_size[1]-(overlap_size[1]*2), :] = result[0][overlap_size[0]:input_size[0]-overlap_size[0], overlap_size[1]:input_size[1]-overlap_size[1]]

  jump = (input_size[0]-(2.25*overlap_size[0]), input_size[1]-(2.25*overlap_size[1]))
  united_images = united_images[jump[0]+overlap_size[0]:visualizer.shape[0]+jump[0]-overlap_size[0], jump[1]+overlap_size[1]:visualizer.shape[1]+jump[1]-overlap_size[0], :]


  result = Image.fromarray((united_images * 255).astype(np.uint8))
  #print im.shape[0]
  result = result.resize((visualizer.shape[1], visualizer.shape[0]), Image.ANTIALIAS)

  result.save(out_path + i[len(path):])
  count = count + 1
  #fig = plt.figure()
  #fig.add_subplot(1,2,1)
  #plt.imshow(np.array(visualizer, dtype=np.uint8))
  #fig.add_subplot(1,2,2)

  #plt.imshow(united_images)
  #plt.show()