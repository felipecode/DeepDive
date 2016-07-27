"""Deep dive libs"""
from input_data_dive_test import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from inception_res_BAC import create_structure
from alex_feature_extract import extract_features

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
from features_on_grid import put_features_on_grid_np
from scipy import misc

import json


"""Verifying options integrity"""
config= configMain()

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    fig.canvas.draw ( )
    w,h = fig.canvas.get_width_height()
    image = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    image.shape = (1, w, h,3 )
    return image

if config.restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')

dataset = DataSetManager(config.training_path, config.validation_path, config.training_path_ground_truth,config.validation_path_ground_truth, config.input_size, config.output_size)
global_step = tf.Variable(0, trainable=False, name="global_step")

""" Creating section"""
x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")
plot_image = tf.placeholder("float", name="output_image_1")
sess = tf.InteractiveSession()
last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout)

" Creating comparation metrics"
y_image = y_
#loss_function = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.pow(tf.sub(last_layer, y_image),2)),3),2),1)
loss_function = tf.reduce_mean(tf.square(tf.sub(last_layer, y_image)))

train_step = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon, use_locking=config.use_locking).minimize(loss_function)


"""Creating summaries"""

tf.image_summary('Input', x)
tf.image_summary('Output', last_layer)
tf.image_summary('GroundTruth', y_image)

for key, l in config.features_list:
 tf.image_summary('Features_map_'+key, put_features_on_grid_tf(feature_maps[key], l))
for key in scalars:
  tf.scalar_summary(key,scalars[key])
for key in config.histograms_list:
 tf.histogram_summary('histograms_'+key, histograms[key])
tf.scalar_summary('Loss', loss_function)

summary_op = tf.merge_all_summaries()
saver = tf.train.Saver(tf.all_variables())

init_op=tf.initialize_all_variables()
sess.run(init_op)
summary_writer = tf.train.SummaryWriter(config.summary_path, graph_def=sess.graph_def)

"""Load a previous model if restore is set to True"""

if not os.path.exists(config.models_path):
  os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)

dados={}
dados['learning_rate']=config.learning_rate
dados['beta1']=config.beta1
dados['beta2']=config.beta2
dados['epsilon']=config.epsilon
dados['use_locking']=config.use_locking
dados['summary_writing_period']=config.summary_writing_period
dados['validation_period']=config.validation_period
dados['batch_size']=config.batch_size
dados['variable_errors']=[]
dados['time']=[]
dados['variable_errors_val']=[]



if config.restore:
  if ckpt:
    print 'Restoring from ', ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
    if os.path.isfile(config.models_path +'summary.json'):
      outfile= open(config.models_path +'summary.json','r+')
      dados=json.load(outfile)
      outfile.close()
    else:
      outfile= open(config.models_path +'summary.json','w')
      json.dump(dados, outfile)
      outfile.close()   
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

training_start_time =time.time()
print config.n_epochs*dataset.getNImagesDataset()/config.batch_size
for i in range(initialIteration, config.n_epochs*dataset.getNImagesDataset()/config.batch_size):
  epoch_number = 1.0+ (float(i)*float(config.batch_size))/float(dataset.getNImagesDataset())

  batch = dataset.train.next_batch(config.batch_size)
  
  """Save the model every 300 iterations"""
  if i%300 == 0:
    saver.save(sess, config.models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'

  start_time = time.time()
  feedDict.update({x: batch[0], y_: batch[1]})
  sess.run(train_step, feed_dict=feedDict)

  duration = time.time() - start_time

  if i%2 == 0:
    examples_per_sec = config.batch_size / duration
    result=sess.run(loss_function, feed_dict=feedDict)
    result > 0
    train_accuracy = result #sum(result)/config.batch_size
    if  train_accuracy < lowest_error:
      lowest_error = train_accuracy
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"%(epoch_number, i, i*config.batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))

  if i%config.summary_writing_period == 1:
    result= sess.run(loss_function, feed_dict=feedDict)#sum(sess.run(loss_function, feed_dict=feedDict))/config.batch_size
    dados['variable_errors'].append(float(result))
    dados['time'].append(time.time() - training_start_time)
    outfile= open(config.models_path +'summary.json','w')
    json.dump(dados, outfile)
    outfile.close()
    if config.use_tensorboard:
      summary_str = sess.run(summary_op, feed_dict=feedDict)
      summary_writer.add_summary(summary_str,i)

  if i%config.validation_period == 0:
    validation_result_error = 0
    for j in range(0,dataset.validation.num_examples/(config.batch_size_val)):
      batch_val = dataset.validation.next_batch(config.batch_size_val)
      feedDictVal = {x: batch_val[0], y_: batch_val[1]}
      validation_result_error+= sess.run(loss_function, feed_dict=feedDictVal) #sum(sess.run(loss_function, feed_dict=feedDictVal))
    dados['variable_errors_val'].append((validation_result_error*config.batch_size_val)/dataset.validation.num_examples)

    outfile= open(config.models_path +'summary.json','w')
    json.dump(dados, outfile)
    outfile.close()   




    #result = Image.fromarray((result[0,:,:,:]*255).astype(np.uint8))
    #result.save(config.validation_path_ground_truth + str(str(i)+ '.jpg'))

    #outfile.write("%f " % (validation_result_error/len(range(0,dataset.validation.num_examples/(config.batch_size_val)) ))  )

    #outfile.write("\n")
    #outfile.close()
  #""" Writing summary, not at every iterations """
  #if i%config.summary_writing_period == 1:
    
    #outfile = open(config.models_path +'variable_errors', 'a+')
    #result= sum(sess.run(loss_function, feed_dict=feedDict))/config.batch_size

    #outfile.write("%f " % result)

    #outfile.write("\n")
    #outfile.close()

    #outfile = open(config.models_path +'time', 'a+')

    #outfile.write("%f\n" % ())
 
    #print iteration
    #print error_vec
    #print val_error_vec

   
    #plot_image = fig2data(figure)
    #print fig2data(figure).dtype

    #addedimage = sess.run(z,feed_dict={plot_image:fig2data(figure)})

    #summary_str = sess.run(summary_op, feed_dict=feedDict)
    #summary_str_val,result 

    #summary_writer.add_summary(summary_str,i)

    

    """ Check here the weights """
    #result = Image.fromarray((result[0,:,:,:]*255).astype(np.uint8))
    #result.save(config.validation_path_ground_truth + str(str(i)+ '.jpg'))
    #summary_writer.add_summary(summary_str_val,i)


   # print config.features_list
   # for key in config.features_list:
   #   print key
   #   print feature_maps
   #   comp_feature_map = sess.run(feature_maps[key], feed_dict= feedDict)
   #   grid_output = put_features_on_grid_np(comp_feature_map)

   #   print grid_output.shape
   #   if not os.path.exists(config.models_path + '/' + key):
   #     os.makedirs(config.models_path + '/' + key)

   #  misc.imsave(config.models_path + '/' + key  + '/' + str(i) + '.png', grid_output)





