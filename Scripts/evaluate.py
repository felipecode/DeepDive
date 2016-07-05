"""Deep dive libs"""
from input_data_dive_test import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
from inception_res_batchSBAC import create_structure

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

config = configMain()

overlap_size = (7, 7)
""" Configuration, set all the variables , including getting all the files that are going to be evaluated. """


path = '/home/adas/DeepDive/Local_results/RealImages/'
out_path ='/home/adas/DeepDive/Local_results/RealImagesResults/'

im_names =  glob.glob(path + "*.jpg")
im_names = im_names + glob.glob(path + "*.png")




print path
print im_names


""" Declare the placeholders variables """

x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")
dout1 = tf.placeholder("float")
dout2 = tf.placeholder("float")
dout3 = tf.placeholder("float")
dout4 = tf.placeholder("float")

# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()
h_conv3, dropoutDict,_,_,_ = create_structure(tf, x, config.input_size,[dout1,dout2,dout3,dout4])
feedDict=dropoutDict

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




count =0
for i in im_names:

  im = Image.open(i).convert('RGB')

  im = np.array(im, dtype=np.float32)
  visualizer = im

  """ Open one image and add some padding to it """

  im = np.lib.pad(im, ((config.input_size[0]-overlap_size[0], 0), (config.input_size[1]-overlap_size[1], 0), (0,0)), mode='constant', constant_values=1)

  """ Initiate all the variables """ 

  original = im
  original = original.astype(np.float32)
  original = np.multiply(original, 1.0 / 255.0)

  height, width = im.shape[0], im.shape[1]

  united_images = np.zeros((im.shape[0]+config.input_size[0], im.shape[1]+config.input_size[1], 3), dtype=np.float32)
  out = np.zeros([config.output_size[0], config.output_size[1], config.output_size[2]]).reshape([1, np.prod(np.array(config.output_size))])


  #print input_size
  print overlap_size


  """Separating the image in chunks"""
  im_vec = []
  #out_vec = [] #  I THINK IT IS NOT NEEDED
  cont =1

  nValues = len(range(0, height, config.input_size[0]-(overlap_size[0]*2)))*len(range(0, width, config.input_size[1]-(overlap_size[1]*2)))
  #print nValues
  res_vec = np.array([])
  for h in range(0, height, config.input_size[0]-(overlap_size[0]*2)):
    for w in range(0, width, config.input_size[1]-(overlap_size[1]*2)):
      h_end = h + config.input_size[0]
      w_end = w + config.input_size[1]


      # Do a more smart cutting to get more chunks at once
      chunk = original[h:h_end, w:w_end]
      if chunk.shape != config.input_size:
        chunk = np.lib.pad(chunk, ((0, config.input_size[0]-chunk.shape[0]), (0, config.input_size[1]-chunk.shape[1]), (0,0)), mode='constant', constant_values=1)
 

      print chunk.shape

      chunk = chunk.reshape([np.prod(np.array(config.input_size))])
      #print chunk.shape
      im_vec.append(chunk.reshape([1, np.prod(np.array(config.input_size))]))
      """ After cutting the image into inputsizes we run the model, the model is run when we extracted enough patches to fill the batch """

      """ Check if the number of read images is equal to the batch size """
      if cont%config.batch_size == 0 or cont == nValues:


        im_vec = np.array(im_vec[:])
        #print im_vec.shape
        im_vec = im_vec.reshape((im_vec.shape[0],config.input_size[0],config.input_size[1],config.input_size[2]))

        print im_vec.shape
        feedDict.update({x: im_vec})
        output = sess.run(h_conv3, feed_dict=feedDict)
       

        #print len(output[0])

        if cont <= config.batch_size:  # It is the first time
          res_vec = output
        else:
          res_vec = np.concatenate([res_vec,output],0)

   
        im_vec = []


      cont = cont +1
      

  cont =0
  """ Rebuild the images"""
  for h in range(0, height, config.input_size[0]-(overlap_size[0]*2 )):
    for w in range(0, width, config.input_size[1]-(overlap_size[1]*2) ):
     
      
      united_images[h:(h+config.input_size[0]-(overlap_size[0]*2 )), w:(w+config.input_size[1]-(overlap_size[1]*2)), :] = res_vec[cont][overlap_size[0]:(config.input_size[0]-overlap_size[0]), overlap_size[1]:(config.input_size[1]-overlap_size[1])] 
      cont = cont +1
      
      #print im.shape
      #print united_images.shape

      
      #print result[0][0].shape



  """ THe invalid Part """      
  jump = (config.input_size[0]-(2.25*overlap_size[0]), config.input_size[1]-(2.25*overlap_size[1]))
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