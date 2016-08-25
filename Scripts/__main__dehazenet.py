"""Deep dive libs"""
from input_data_levelDB import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
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
from ssim_tf import ssim_tf
from utils import *

"""Verifying options integrity"""
config = configDehazenet()

#depois a gente coloca isso no config
#o segundo parametro e o numero de linhas pra mostrar


if config.restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')

dataset = DataSetManager(config.training_path, config.validation_path, config.training_path_ground_truth, 
                         config.validation_path_ground_truth, config.input_size, config.output_size)
global_step = tf.Variable(0, trainable=False, name="global_step")


""" Creating section"""
x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")
sess = tf.InteractiveSession()
last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout)

" Creating comparation metrics"
y_image = y_
loss_function = tf.abs(tf.sub(last_layer,  tf.reduce_mean(y_image,reduction_indices=[1,2])))
loss_mean = tf.reduce_mean(tf.abs(tf.sub(last_layer,  tf.reduce_mean(y_image,reduction_indices=[1,2]))))
# using the same function with a different name
#print last.shape
#loss_function_ssim = ssim_tf(tf,y_image,last_layer)

train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_function)

"""Creating summaries"""

tf.image_summary('Input', x)
tf.image_summary('Output', last_layer)
#tf.image_summary('GroundTruth', y_)

for key, l in config.features_list:
 tf.image_summary('Features_map_'+key, put_features_on_grid_tf(feature_maps[key], l))
for key in scalars:
  tf.scalar_summary(key,scalars[key])
for key in config.histograms_list:
 tf.histogram_summary('histograms_'+key, histograms[key])
tf.scalar_summary('Loss', loss_function)
#tf.scalar_summary('Loss_SSIM', loss_function_ssim)

summary_op = tf.merge_all_summaries()
saver = tf.train.Saver(tf.all_variables())

val = tf.scalar_summary('Loss_Validation', loss_function)
init_op = tf.initialize_all_variables()
sess.run(init_op)
summary_writer = tf.train.SummaryWriter(config.summary_path,
                                            graph_def=sess.graph_def)


"""Load a previous model if restore is set to True"""

if not os.path.exists(config.models_path):
  os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)
print ckpt
if config.restore:
  if ckpt.model_checkpoint_path:
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


error_vec = []
val_error_vec = []
iteration = []
iteration_val = []
tmp_queue = []
begin = time.time()
for i in range(initialIteration, config.n_epochs*dataset.getNImagesDataset()/config.batch_size):
  if(i%dataset.getNImagesDataset()/config.batch_size == 0):
    print time.time() - begin
    begin = time.time()


  epoch_number = 1.0 + (float(i)*float(config.batch_size))/float(dataset.getNImagesDataset())


  
  """ Do validation error and generate Images """
  batch = dataset.train.next_batch(config.batch_size)
  
  """Save the model every 300 iterations"""
  if i%300 == 0:
    saver.save(sess, config.models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'

  start_time = time.time()

  feedDict.update({x: batch[0], y_: batch[1]})
  sess.run(train_step, feed_dict=feedDict)
  
  duration = time.time() - start_time

  if i%20 == 0:
    num_examples_per_step = config.batch_size 
    examples_per_sec = num_examples_per_step / duration
    train_accuracy = sess.run(loss_function, feed_dict=feedDict)
    if  np.mean(train_accuracy) < lowest_error:
      lowest_error = np.mean(train_accuracy)
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"
        %(epoch_number, i, i*config.batch_size, np.mean(train_accuracy), lowest_error, lowest_iter,examples_per_sec))

  """ Writing summary, not at every iterations """
  if i%20 == 0:
    result = sess.run(loss_mean, feed_dict=feedDict)
    tmp_queue.append(result)
    if len(tmp_queue) > 20:
      tmp_queue.pop(0)

    iteration.append(i)
    error_vec.append(np.mean(tmp_queue))

    """
    batch_val = dataset.validation.next_batch(config.batch_size)
    summary_str = sess.run(summary_op, feed_dict=feedDict)
    #summary_str_val,result= sess.run([val,last_layer], feed_dict=feedDict)
    feedDict.update({x: batch_val[0], y_: batch_val[1]})
    summary_str_val,result= sess.run([val,last_layer], feed_dict=feedDict)
    validation_accuracy = sess.run(loss_validation, feed_dict = feedDict)
#    print summary_str_val
    print "validation result"
    print abs(result - batch_val[1])
    #print np.mean(np.mean(batch[1],axis=1),axis=1)
#    print batch[1].shape
    summary_writer.add_summary(summary_str,i)

    #result = Image.fromarray((result[0,:,:,:]*255).astype(np.uint8))
    #result.save(config.validation_path_ground_truth + str(str(i)+ '.jpg'))
    summary_writer.add_summary(summary_str_val,i)
  """

  if i%5000 == 1:

    print ' VALIDATING '
    iteration_val.append(i)
    summary_str_val = 0
    for j in range(1,dataset.getNImagesValidation()/(8*config.batch_size_val)):
      batch_val = dataset.validation.next_batch(config.batch_size_val)

      #print feature_maps
      summary_str_val +=  sess.run(loss_mean, feed_dict={x: batch_val[0], y_: batch_val[1]})
      #print j

    val_error_vec.append(summary_str_val/len(range(1,dataset.getNImagesValidation()/(8*config.batch_size_val))))
    figure = plt.figure(  )
    plot   = figure.add_subplot ( 111 )

    #for key, l in config.features_list

    plot.plot(iteration, error_vec, 'b-', iteration_val, val_error_vec, 'r-')
    figure.savefig(config.models_path + str(i) + '.png')
    plt.close(figure)
