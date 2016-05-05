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
 images[0] = img_noise.copy()
 step_size=config.opt_step
 print("Maximizing output of channel %d of layer %s"%(channel, key))
 opt_name='Opt_'+key+"_"+str(channel)
 opt_summary=tf.image_summary(opt_name, x)
 ft_summary=tf.image_summary(key+"_"+str(channel), tf.expand_dims(feature_maps[key][:,:,:,channel],3))
 summary_op=tf.merge_summary([opt_summary,ft_summary,tf.image_summary('Output_'+opt_name, last_layer)])
 for i in xrange(1,config.opt_iter_n+1):
  feedDict.update({x: images, y_: images})
  g, score = sess.run([t_grad, t_score], feed_dict=feedDict)
  # normalizing the gradient, so the same step size should work for different layers and networks
  g /= g.std()+1e-8
  images[0] = images[0]+g*step_size
  if config.l2_decay:
   images[0] = images[0]*(1-config.decay)
  if config.gaussian_blur:
   if i%config.blur_iter==0:
    images[0] = gaussian_filter(images[0], sigma=config.blur_width)
  if config.clip_norm:
   norms=np.linalg.norm(images[0], axis=2, keepdims=True)
   n_thrshld=np.sort(norms, axis=None)[int(norms.size*config.norm_pct_thrshld)]
   images[0]=images[0]*(norms>n_thrshld)
  if config.clip_contrib:
   contribs=np.sum(images[0]*g[0], axis=2, keepdims=True)
   c_thrshld=np.sort(contribs, axis=None)[int(contribs.size*config.contrib_pct_thrshld)]
   images[0]=images[0]*(contribs>c_thrshld)

  if i%10 == 0:
    print("Step %d, score of channel %d of layer %s: %f"%(i, channel, key, score))
    summary_str = sess.run(summary_op, feed_dict=feedDict)
    summary_writer.add_summary(summary_str,i)
