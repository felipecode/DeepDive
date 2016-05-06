"""Deep dive libs"""
from input_data_dive_test import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from depth_map_structure_dropout2 import create_structure

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
from scipy.ndimage.filters import gaussian_filter

from features_on_grid import put_features_on_grid

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    std = tf.sqrt(tf.reduce_mean(tf.square(img)))
    return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    k = np.float32([1,4,6,4,1])
    k = np.outer(k, k)
    k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
    img = tf.expand_dims(img,0)

    levels = []
    for i in xrange(scale_n):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
        levels.append(hi)
	img=lo
    levels.append(img)
    tlevels=levels[::-1]
    tlevels = map(normalize_std, tlevels)

    img = tlevels[0]
    for hi in tlevels[1:]:
        img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img[0,:,:,:]

"""Verifying options integrity"""
config= configOptimization()

if config.restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')


""" Creating section"""
x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")
sess = tf.InteractiveSession()
last_layer, dropoutDict, feature_maps,_ = create_structure(tf, x,config.input_size,config.dropout)

saver = tf.train.Saver(tf.all_variables())

summary_writer = tf.train.SummaryWriter(config.summary_path,
                                            graph=sess.graph)

"""Load a previous model if restore is set to True"""

if not os.path.exists(config.models_path):
  os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)
if config.restore:
  if ckpt:
    print 'Restoring from ', ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
  ckpt = 0

print 'Logging into ' + config.summary_path

feedDict=dropoutDict

images = np.empty((1, config.input_size[0], config.input_size[1], config.input_size[2]))
img_noise = np.random.uniform(low=0.0, high=1.0, size=config.input_size)


for key, channel in config.features_opt_list:
 t_score = tf.reduce_mean(feature_maps[key][:,:,:,channel])
 t_grad = tf.gradients(t_score, x)[0]

 if config.lap_grad_normalization:
  grad_norm=lap_normalize(t_grad[0,:,:,:])
 else:
  grad_norm=normalize_std(t_grad)

 images[0] = img_noise.copy()
 step_size=config.opt_step
 print("Maximizing output of channel %d of layer %s"%(channel, key))
 opt_name='Opt_'+key+"_"+str(channel)
 opt_summary=tf.image_summary(opt_name, x)
 ft_summary=tf.image_summary(key+"_"+str(channel), tf.expand_dims(feature_maps[key][:,:,:,channel],3))
 summary_op=tf.merge_summary([opt_summary,ft_summary,tf.image_summary('Output_'+opt_name, last_layer)])
 for i in xrange(1,config.opt_iter_n+1):
  feedDict.update({x: images, y_: images})
  g, score = sess.run([grad_norm, t_score], feed_dict=feedDict)
  # normalizing the gradient, so the same step size should work for different layers and networks
  images[0] = images[0]+g*step_size
  #l2 decay
  images[0] = images[0]*(1-config.decay)
  #gaussian blur
  if config.blur_iter:
   if i%config.blur_iter==0:
    images[0] = gaussian_filter(images[0], sigma=config.blur_width)
  #clip norm
  norms=np.linalg.norm(images[0], axis=2, keepdims=True)
  n_thrshld=np.sort(norms, axis=None)[int(norms.size*config.norm_pct_thrshld)]
  images[0]=images[0]*(norms>=n_thrshld)
  #clip contribution
  contribs=np.sum(images[0]*g[0], axis=2, keepdims=True)
  c_thrshld=np.sort(contribs, axis=None)[int(contribs.size*config.contrib_pct_thrshld)]
  images[0]=images[0]*(contribs>=c_thrshld)

  if i%10 == 0:
    print("Step %d, score of channel %d of layer %s: %f"%(i, channel, key, score))
    summary_str = sess.run(summary_op, feed_dict=feedDict)
    summary_writer.add_summary(summary_str,i)
