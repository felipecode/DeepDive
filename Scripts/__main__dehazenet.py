"""Deep dive libs"""
from input_data_dive_test import DataSetManager
from config_dehazenet import *

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



"""Verifying options integrity"""

if restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')

dataset = DataSetManager(path, val_path, input_size, proportions)
global_step = tf.Variable(0, trainable=False, name="global_step")

x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")

sess = tf.InteractiveSession()

last_layer, l2_reg = create_structure(tf, x,input_size,[])

y_image = y_

loss_function = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)))

# using the same function with a different name
loss_validation = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(last_layer, tf.reduce_mean(y_image)),2)),name='Validation')
print y_image
print last_layer

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

"""Creating summaries"""
tf.image_summary('Input', x)
tf.image_summary('Output', last_layer)
tf.image_summary('GroundTruth', y_)
tf.scalar_summary('Loss', loss_function)
summary_op = tf.merge_all_summaries()

saver = tf.train.Saver(tf.all_variables())

val  = tf.scalar_summary('Loss_Validation', loss_validation)

sess.run(tf.initialize_all_variables())


summary_writer = tf.train.SummaryWriter(summary_path,  graph_def=sess.graph_def)
  
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


if ckpt:
  initialIteration = int(ckpt.model_checkpoint_path.split('-')[1])
else:
  initialIteration = 1

for i in range(initialIteration, n_epochs*n_images_dataset):

  
  epoch_number = 1.0+ (float(i)*float(batch_size))/float(n_images_dataset)
 

  """Save the model every 300 iterations"""
  if i%300 == 0:
    # if ckpt:
    #   saver.save(sess, models_path + 'model.ckpt', global_step=i + int(ckpt.model_checkpoint_path.split('-')[1]))
    #   print 'Model saved.'
    # else:
    saver.save(sess, models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'

  start_time = time.time()

  """ Do validation error and generate Images """  
  batch = dataset.train.next_batch(batch_size)

  """Calculate the loss"""
  #train_accuracy = loss_function.eval(feed_dict={
  #    x:batch[0], y_: batch[1]})
  """Run training and write the summaries"""
  #train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

  duration = time.time() - start_time

  if i%20 == 0:
    num_examples_per_step = batch_size 
    examples_per_sec = num_examples_per_step / duration
    train_accuracy = sess.run(loss_function, feed_dict={x: batch[0], y_: batch[1]})
    if  train_accuracy < lowest_error:
      lowest_error = train_accuracy
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"%(epoch_number, i, i*batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))
 

  #if ckpt:
  #  summary_writer.add_summary(summary_str, i + int(ckpt.model_checkpoint_path.split('-')[1]))
  #else:
  """ Writing summary, not at every iterations """
  if i%20 == 0:
    batch_val = dataset.validation.next_batch(batch_size)
    summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1]})
    summary_str_val, result = sess.run([val,last_layer], feed_dict={x: batch_val[0], y_: batch_val[1]})
    summary_writer.add_summary(summary_str, i)
    summary_writer.add_summary(summary_str_val, i)

    """ Check here the weights """
    #print result

    # @BUG dando falha quando a imagem vai ser salva na linha abaixo, por que?
    #result = Image.fromarray((result[0,:,:,:]*255).astype(np.uint8))
    #result.save(out_path + str(str(i)+ '.jpg'))
  

