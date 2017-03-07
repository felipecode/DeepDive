"""Deep dive libs"""
from input_data_levelDB import DataSetManager
from config import *
from utils import *
from features_optimization import optimize_feature
from loss_network import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from inception_res_BAC_normalized import create_structure

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
from scipy import misc

import json


"""Verifying options integrity"""
config = configMain()

if config.restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')
if config.save_features_to_disk not in (True, False):
  raise Exception('Wrong save_features_to_disk option. (True or False)')
if config.save_json_summary not in (True, False):
  raise Exception('Wrong save_json_summary option. (True or False)') 
if config.use_tensorboard not in (True, False):
  raise Exception('Wrong use_tensorboard option. (True or False)')

dataset = DataSetManager(config) 
global_step = tf.Variable(0, trainable=False, name="global_step")

""" Creating section"""
x = tf.placeholder("float",(config.batch_size,)+config.input_size, name="input_image")
y_ = tf.placeholder("float",(config.batch_size,)+config.output_size, name="output_image")
lr = tf.placeholder("float", name = "learning_rate")
#training = tf.placeholder(tf.bool, name="training")


sess = tf.InteractiveSession()
last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout)

feature_loss=create_loss_structure(tf, 255.0*last_layer, 255.0*y_, sess)

" Creating comparison metrics"
y_image = y_
#lab_mse_loss = tf.reduce_mean(np.absolute(np.subtract(color.rgb2lab((255.0*last_layer).eval()), color.rgb2lab((255.0*y_image).eval()))))
mse_loss = tf.reduce_mean(tf.squared_difference(255.0*last_layer, 255.0*y_image), reduction_indices=[1,2,3]) 
loss_function = feature_loss#mse_loss #(mse_loss+feature_loss)/2

#loss_function = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.pow(tf.sub(last_layer, y_image),2)),3),2),1)
#loss_function = tf.reduce_mean(tf.abs(tf.sub(last_layer, y_image)))
print y_image
train_step = tf.train.AdamOptimizer(learning_rate = lr, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon,
                                    use_locking=config.use_locking).minimize(loss_function)


"""Creating summaries"""

tf.image_summary('Input', x)
tf.image_summary('Output', last_layer)
tf.image_summary('GroundTruth', y_image)



# for key in config.features_list:
#   ft_ops.append(feature_maps[key])
ft_ops=[]
weights=[]
for key in config.features_list:
  ft_ops.append(feature_maps[key][0])
  weights.append(feature_maps[key][1])
for key in scalars:
  tf.scalar_summary(key,scalars[key])
for key in config.histograms_list:
 tf.histogram_summary('histograms_'+key, histograms[key])
tf.scalar_summary('Loss', tf.reduce_mean(loss_function))
tf.scalar_summary('feature_loss',tf.reduce_mean(feature_loss))
tf.scalar_summary('mse_loss',tf.reduce_mean(mse_loss))
tf.scalar_summary('learning_rate',lr)


summary_op = tf.merge_all_summaries()
saver = tf.train.Saver(tf.all_variables())

init_op=tf.initialize_all_variables()
sess.run(init_op)

summary_writer = tf.train.SummaryWriter(config.summary_path, graph=sess.graph)

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
dados['learning_rate_update']=[]
if config.restore:
  if ckpt:
    print 'Restoring from ', ckpt.model_checkpoint_path
    saver.restore(sess,ckpt.model_checkpoint_path)
    if config.save_json_summary:
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

lowest_error = 1500;
lowest_val  = 1500;
lowest_iter = 1;
lowest_val_iter = 1;

feedDict=dropoutDict
if ckpt:
  tamanho=len(ckpt.model_checkpoint_path.split('-'))
  initialIteration = int(ckpt.model_checkpoint_path.split('-')[tamanho-1])
else:
  initialIteration = 1

training_start_time =time.time()

print "q", config.n_epochs*dataset.getNImagesDataset()/config.batch_size
for i in range(initialIteration, config.n_epochs*dataset.getNImagesDataset()/config.batch_size):
  epoch_number = (float(i)*float(config.batch_size))/float(dataset.getNImagesDataset())
  """Save the model every 300 iterations"""
  if i%300 == 0:
    saver.save(sess, config.models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'


  start_time = time.time()

  batch = dataset.train.next_batch(config.batch_size)
  feedDict.update({x: batch[0], y_: batch[1], lr: (config.learning_rate/(config.lr_update_value ** int(int(epoch_number)/config.lr_update_period)))})
  sess.run(train_step, feed_dict=feedDict)

  duration = time.time() - start_time


  if i%4 == 0:
    examples_per_sec = config.batch_size / duration
    result=sess.run(loss_function, feed_dict=feedDict)
    result > 0
    train_accuracy = sum(result)/config.batch_size
    if  train_accuracy < lowest_error:
      lowest_error = train_accuracy
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"
        %(epoch_number, i, i*config.batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))

  if i%config.summary_writing_period == 1 and (config.use_tensorboard or config.save_features_to_disk or config.save_json_summary):
    output, result = sess.run([last_layer,loss_function], feed_dict=feedDict)
    result = np.mean(result)
    if len(ft_ops) > 0:
      ft_maps= sess.run(ft_ops, feed_dict=feedDict)
    else:
      ft_maps= []

    if config.use_deconv:
	deconv=deconvolution(x, feedDict, ft_ops, config.features_list, config.batch_size, config.input_size)
    else:
    	deconv=[None]*len(ft_ops)

    if config.save_json_summary:
      dados['variable_errors'].append(float(result))
      dados['time'].append(time.time() - training_start_time)
      outfile = open(config.models_path +'summary.json','w')
      json.dump(dados, outfile)
      outfile.close()
    if config.use_tensorboard:
      summary_str = sess.run(summary_op, feed_dict=feedDict)
      summary_writer.add_summary(summary_str,i)
      if len(ft_ops) > 0:
        for ft, w, d, key in zip(ft_maps, weights, deconv, config.features_list):
         ft_grid=put_features_on_grid_np(ft)
         ft_name="Features_map_"+key
         ft_summary=tf.image_summary(ft_name, ft_grid)
         summary_str=sess.run(ft_summary)
         summary_writer.add_summary(summary_str,i)
	 if w is not None:
	 	kernel=w.eval()
	 	kernel_grid=put_kernels_on_grid_np(kernel)
	 	kernel_name="kernels_"+key
	 	kernel_summary=tf.image_summary(kernel_name, kernel_grid)
	 	kernel_summary_str=sess.run(kernel_summary)
	 	summary_writer.add_summary(kernel_summary_str,i)
	 if d is not None:
		deconv_grid=put_grads_on_grid_np(d.astype(np.float32))
		deconv_name="deconv_"+key
		deconv_summary=tf.image_summary(deconv_name, deconv_grid)
	 	deconv_summary_str=sess.run(deconv_summary)
	 	summary_writer.add_summary(deconv_summary_str,i)

    if(config.save_features_to_disk):
      save_images_to_disk(output,batch[0],batch[1],config.summary_path)
      save_feature_maps_to_disk(ft_maps, weights, deconv, config.features_list,config.summary_path)

  if i%config.validation_period == 0:
    error_per_transmission=[0.0] * config.num_bins
    count_per_transmission=[0] * config.num_bins
    validation_result_error = 0
    for j in range(0,dataset.getNImagesValidation()/(config.batch_size)):
      batch_val = dataset.validation.next_batch(config.batch_size)
      feedDictVal = {x: batch_val[0], y_: batch_val[1]}
      result = sess.run(loss_function, feed_dict=feedDictVal)
      validation_result_error += sum(result)
      if config.save_error_transmission:
        for i in range(len(batch_val[2])):
          index = int(float(batch_val[2][i]) * config.num_bins)
          error_per_transmission[index] += result[i]
          count_per_transmission[index] += 1
        for i in range(config.num_bins):
          if count_per_transmission[i]!=0:
            error_per_transmission[i] = error_per_transmission[i]/count_per_transmission[i]
        dados['error_per_transmission']=error_per_transmission

    if dataset.getNImagesValidation() !=0 :
      validation_result_error = (validation_result_error)/dataset.getNImagesValidation()
      
    if config.use_tensorboard:
      val=tf.scalar_summary('Loss_Validation', validation_result_error)
      summary_str_val=sess.run(val)
      summary_writer.add_summary(summary_str_val,i)
    if config.save_json_summary:
      dados['variable_errors_val'].append(validation_result_error)
      outfile= open(config.models_path +'summary.json','w')
      json.dump(dados, outfile)
      outfile.close()

  if config.opt_every_iter>0 and i%config.opt_every_iter==0:
    """ Optimization """
    print("Running Optimization")
    for key, channel in config.features_opt_list:
        ft=feature_maps[key][0]
        n_channels=ft.get_shape()[3]
        if channel<0:
          #otimiza todos os canais
          for ch in xrange(n_channels):
            opt_output=optimize_feature(config.input_size, x, ft[:,:,:,ch])
            if config.use_tensorboard:
              opt_name="optimization_"+key+"_"+str(ch).zfill(len(str(n_channels)))
              opt_summary=tf.image_summary(opt_name, np.expand_dims(opt_output,0))
              summary_str=sess.run(opt_summary)
              summary_writer.add_summary(summary_str,i)
          # salvando as imagens como bmp
            if(config.save_features_to_disk):
              save_optimazed_image_to_disk(opt_output,ch,n_channels,key,config.summary_path)
        else:
          opt_output=optimize_feature(config.input_size, x, ft[:,:,:,channel])
          if config.use_tensorboard:
            opt_name="optimization_"+key+"_"+str(channel).zfill(len(str(n_channels)))
            opt_summary=tf.image_summary(opt_name, np.expand_dims(opt_output,0))
            summary_str=sess.run(opt_summary)
            summary_writer.add_summary(summary_str,i)
          # salvando as imagens como bmp
          if(config.save_features_to_disk):
            save_optimazed_image_to_disk(opt_output,channel,n_channels,key,config.summary_path)
