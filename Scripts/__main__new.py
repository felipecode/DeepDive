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
from math import log10

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
from features_on_grid import put_features_on_grid
from features_optimization import optimize_feature

"""Verifying options integrity"""
config= configMain()


if config.restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')

dataset = DataSetManager(config.training_path, config.validation_path, config.training_path_ground_truth,config.validation_path_ground_truth, config.input_size, config.output_size)
global_step = tf.Variable(0, trainable=False, name="global_step")


""" Creating section"""
x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")
sess = tf.InteractiveSession()
last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout)

" Creating comparation metrics"
y_image = y_
loss_function = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)))
# using the same function with a different name
loss_validation = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)),name='Validation')
#loss_function_ssim = ssim_tf(tf,y_image,last_layer)

train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_function)

"""Creating summaries"""

tf.image_summary('Input', x)
tf.image_summary('Output', last_layer)
tf.image_summary('GroundTruth', y_)
#tf.image_summary('Output', tf.reshape(tf.image.grayscale_to_rgb(last_layer),[16,16,3,1]))
#tf.image_summary('GroundTruth', tf.reshape(tf.image.grayscale_to_rgb(y_),[16,16,3,1]))

#test = tf.get_default_graph().get_tensor_by_name("scale_1/Scale1_first_relu:0")
#tf.image_summary('Teste', put_features_on_grid(test, 8))
for key, l in config.features_list:
 tf.image_summary('Features_map_'+key, put_features_on_grid(feature_maps[key], l))
for key in scalars:
  tf.scalar_summary(key,scalars[key])
for key in config.histograms_list:
  tf.histogram_summary('histograms_'+key, histograms[key])
tf.scalar_summary('Loss', loss_function)
#tf.scalar_summary('Loss_SSIM', loss_function_ssim)

summary_op = tf.merge_all_summaries()
saver = tf.train.Saver(tf.all_variables())

val  =tf.scalar_summary('Loss_Validation', loss_validation)

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

for i in range(initialIteration, config.n_epochs*dataset.getNImagesDataset()):

  
  epoch_number = 1.0+ (float(i)*float(config.batch_size))/float(dataset.getNImagesDataset())


  
  """ Do validation error and generate Images """
  batch = dataset.train.next_batch(config.batch_size)
  
  """Save the model every 300 iterations"""
  if i%300 == 0:
    saver.save(sess, config.models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'

  start_time = time.time()
  #print batch[0].shape
  feedDict.update({x: batch[0], y_: batch[1]})
  sess.run(train_step, feed_dict=feedDict)
  
  duration = time.time() - start_time

  if i%20 == 0:
    num_examples_per_step = config.batch_size 
    examples_per_sec = num_examples_per_step / duration
    train_accuracy = sess.run(loss_function, feed_dict=feedDict)
    if  train_accuracy < lowest_error:
      lowest_error = train_accuracy
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"%(epoch_number, i, i*config.batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))

  """ Writing summary, not at every iterations """
  if i%20 == 0:
    batch_val = dataset.validation.next_batch(config.batch_size)
    summary_str = sess.run(summary_op, feed_dict=feedDict)
    summary_str_val,result= sess.run([val,last_layer], feed_dict=feedDict)
    summary_writer.add_summary(summary_str,i)

    """ Check here the weights """
    #result = Image.fromarray((result[0,:,:,:]*255).astype(np.uint8))
    #result.save(config.validation_path_ground_truth + str(str(i)+ '.jpg'))
    #summary_writer.add_summary(summary_str_val,i)
  
  if config.opt_every_iter>0 and i%config.opt_every_iter==0:
    """ Optimization """
    print("Running Optimization")
    for key, channel in config.features_opt_list:
     ft=feature_maps[key]
     n_channels=ft.get_shape()[3]
     if channel<0:
      #otimiza todos os canais       
      for ch in xrange(n_channels):
	output=optimize_feature(config.input_size, x, ft[:,:,:,ch])
	opt_name="optimization_"+key+"_"+str(ch).zfill(len(str(n_channels)))
	opt_summary=tf.image_summary(opt_name, np.expand_dims(output,0))
	summary_str=sess.run(opt_summary)
	summary_writer.add_summary(summary_str,i)
     else:
      output=optimize_feature(config.input_size, x, ft[:,:,:,channel])
      opt_name="optimization_"+key+"_"+str(channel).zfill(len(str(n_channels)))
      opt_summary=tf.image_summary(opt_name, np.expand_dims(output,0))
      summary_str=sess.run(opt_summary)
      summary_writer.add_summary(summary_str,i)
