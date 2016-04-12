"""Deep dive libs"""
from input_data_dive_test import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from dehazenet_structure import create_structure

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



def put_kernels_on_grid (kernel, (grid_Y, grid_X), pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x8, dtype=tf.uint8)




"""Verifying options integrity"""

if restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')

dataset = DataSetManager(path, val_path, pathGroundTruth, out_path, input_size,output_size, proportions)
global_step = tf.Variable(0, trainable=False, name="global_step")

x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")

sess = tf.InteractiveSession()

last_layer, l2_reg = create_structure(tf, x, input_size, [])

y_image = y_

loss_function = tf.reduce_mean(tf.sqrt(tf.pow(tf.sub(last_layer, tf.reduce_mean(y_image)),2)))
# using the same function with a different name
loss_validation = tf.reduce_mean(tf.sqrt(tf.pow(tf.sub(last_layer, tf.reduce_mean(y_image)),2),name='Validation'))




train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

"""Creating summaries"""
tf.image_summary('Input', x)

tf.scalar_summary('Loss', loss_function)

summary_op = tf.merge_all_summaries()

saver = tf.train.Saver(tf.all_variables())

val  =tf.scalar_summary('Loss_Validation', loss_validation)

sess.run(tf.initialize_all_variables())

summary_writer = tf.train.SummaryWriter(summary_path, graph_def=sess.graph_def)

"""Load a previous model if restore is set to True"""

if not os.path.exists(models_path):
  os.mkdir(models_path)
ckpt = tf.train.get_checkpoint_state(models_path)

print ckpt
if restore:
  if ckpt.model_checkpoint_path:
    print 'Restoring from ', ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
  ckpt = 0


print 'Logging into ' + summary_path



"""Training"""

lowest_error = 1.5;
lowest_val  = 1.5;
lowest_iter = 1;
lowest_val_iter = 1;

#i + int(ckpt.model_checkpoint_path.split('-')[1])


if ckpt:
  initialIteration = int(ckpt.model_checkpoint_path.split('-')[1])
else:
  initialIteration = 1

for i in range(initialIteration, n_epochs*n_images_dataset):

  
  epoch_number = 1.0+ (float(i)*float(batch_size))/float(n_images_dataset)


  
  """ Do validation error and generate Images """
  batch = dataset.train.next_batch(batch_size)
  """Save the model every 300 iterations"""
  if i%300 == 0:
    saver.save(sess, models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'


  start_time = time.time()
  """Calculate the loss"""
  """Run training and write the summaries"""
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})  
  duration = time.time() - start_time

  if i%20 == 0:
    num_examples_per_step = batch_size 
    examples_per_sec = num_examples_per_step / duration
    train_accuracy = sess.run(loss_function, feed_dict={x: batch[0], y_: batch[1]})
    print train_accuracy
    if  train_accuracy < lowest_error:
      lowest_error = train_accuracy
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"%(epoch_number, i, i*batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))

  """ Writing summary, not at every iterations """
  if i%20 == 0:
    batch_val = dataset.validation.next_batch(batch_size)
    summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1]})
    summary_str_val,result= sess.run([val,last_layer], feed_dict={x: batch_val[0], y_: batch_val[1]})
    summary_writer.add_summary(summary_str,i)
    print sess.run(last_layer,feed_dict={x: batch[0]})

    """ Check here the weights """
    summary_writer.add_summary(summary_str_val,i)
  print 

