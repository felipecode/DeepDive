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

#dataset = DataSetManager(config.training_path, config.validation_path, config.training_path_ground_truth,config.validation_path_ground_truth, config.input_size, config.output_size)
global_step = tf.Variable(0, trainable=False, name="global_step")


""" Creating section"""
x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")
sess = tf.InteractiveSession()
last_layer, dropoutDict, feature_maps,scalars = create_structure(tf, x,config.input_size,config.dropout)

" Creating comparation metrics"
y_image = y_
#loss_function = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)))
# using the same function with a different name
#loss_validation = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)),name='Validation')
#loss_function_ssim = ssim_tf(tf,y_image,last_layer)

#train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_function)

"""Creating summaries"""

tf.image_summary('Output', last_layer)
#tf.image_summary('GroundTruth', y_)

#test = tf.get_default_graph().get_tensor_by_name("scale_1/Scale1_first_relu:0")
#tf.image_summary('Teste', put_features_on_grid(test, 8))
for key, l in config.features_list:
 tf.image_summary('Features_map_'+key, put_features_on_grid(feature_maps[key], l))
#for key in scalars:
#  tf.scalar_summary(key,scalars[key])
#tf.scalar_summary('Loss', loss_function)
#tf.scalar_summary('Loss_SSIM', loss_function_ssim)

summary_op = tf.merge_all_summaries()
saver = tf.train.Saver(tf.all_variables())

#val  =tf.scalar_summary('Loss_Validation', loss_validation)

sess.run(tf.initialize_all_variables())

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

"""Training"""

lowest_error = 1.5;
lowest_val  = 1.5;
lowest_iter = 1;
lowest_val_iter = 1;

feedDict=dropoutDict
if ckpt:
  initialIteration = int(ckpt.model_checkpoint_path.split('-')[1])
else:
  initialIteration = 1

images = np.empty((1, config.input_size[0], config.input_size[1], config.input_size[2]))
img_noise = np.random.uniform(low=0.0, high=1.0, size=config.input_size)

features_opt={}
for key, channel in config.features_opt_list:
 t_score = tf.reduce_mean(feature_maps[key][:,:,:,channel])
 t_grad = tf.gradients(t_score, x)[0]  
 images[0] = img_noise.copy()
 step_size=config.opt_step
 print("Maximizing output of channel %d of layer %s"%(channel, key))
 opt_name='Optimization_'+key+"_"+str(channel)
 optm_summary=tf.image_summary(opt_name, x)
 for i in xrange(config.opt_iter_n):
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
  images[0]=np.minimum(1,images[0])
  images[0]=np.maximum(0,images[0])
#  images[0]-=min(0,np.amin(images[0]))
#  images[0]=images[0]/(max(1,np.amax(images[0])-min(0,np.amin(images[0]))))

  if i%100 == 0:
    print("Step %d, score of channel %d of layer %s: %f"%(i, channel, key, score))
    summary_opt_str, summary_str = sess.run([optm_summary, summary_op], feed_dict=feedDict)
    summary_writer.add_summary(summary_str,i)
    summary_writer.add_summary(summary_opt_str,i)
