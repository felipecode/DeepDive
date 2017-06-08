#-*- encoding:UTF-8 -*-
#pylint: disable=W0311
"""Deep dive libs"""
from input_data_levelDB_escape_data_augmentation import DataSetManager
from config import *
from utils import *
from loss_network import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from escapenet import create_structure

"""Core libs"""
import tensorflow as tf
import numpy as np


"""Python libs"""
import os
import time

import json

tf.device('/gpu:0')
"""Verifying options integrity"""
def verifyConfig(config):
  if config.restore not in (True, False):
    raise Exception('Wrong restore option. (True or False)')
  if config.save_features_to_disk not in (True, False):
    raise Exception('Wrong save_features_to_disk option. (True or False)')
  if config.save_json_summary not in (True, False):
    raise Exception('Wrong save_json_summary option. (True or False)') 
  if config.use_tensorboard not in (True, False):
    raise Exception('Wrong use_tensorboard option. (True or False)')
  if config.use_locking not in (True, False):
    raise Exception('Wrong use_locking option. (True or False)')
  if config.use_deconv not in (True, False):
    raise Exception('Wrong use_deconv option. (True or False)')
config = configEscape()
verifyConfig(config)

"""Creating session"""
sess = tf.InteractiveSession()#config=tf.ConfigProto(log_device_placement=True))
dataset = DataSetManager(config)
print dataset.getNImagesDataset()

global_step = tf.Variable(0, trainable=False, name="global_step")

"""Creating Placeholders"""
batch_size=config.batch_size
tf_images=tf.placeholder("float",(batch_size,) +config.input_size, name="images")  #inputs
tf_points=tf.placeholder("float",(batch_size,) +config.output_size, name="points")  #ground truth
lr = tf.placeholder("float", name = "learning_rate")
x = tf_images
with tf.variable_scope("network", reuse=None):
  last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout)

network_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
l2_loss = (tf.sqrt(tf.reduce_sum(tf.square(tf_points - last_layer), 1)))
print l2_loss
#l1_loss = tf.reduce_mean((tf.reduce_sum(tf.abs(tf_points - last_layer), 1)))
loss_function = l2_loss
train_step = tf.train.AdamOptimizer(learning_rate = lr, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon,
                                    use_locking=config.use_locking).minimize(loss_function, var_list=network_vars)
"""Creating summaries"""

#tf.image_summary('Input', x)
#tf.image_summary('Output', last_layer)TODO: Gerar imagens pra visualização
#tf.image_summary('GroundTruth', y_image)

tf.summary.scalar('GroundTruthX', tf_points[0,0,0,0])
tf.summary.scalar('OutputX', last_layer[0,0,0,0])

tf.summary.scalar('GroundTruthY', tf_points[0,1,0,0])
tf.summary.scalar('OutputY', last_layer[0,1,0,0])
tf.summary.scalar('Loss', tf.reduce_mean(loss_function))
tf.summary.scalar('learning_rate',lr)



ft_ops=[]
weights=[]
summary_op = tf.summary.merge_all()

val_error = tf.placeholder(tf.float32, shape=(), name="Validation_Error")
val_summary=tf.summary.scalar('Loss_Validation', val_error)

init_op=tf.initialize_all_variables()
sess.run(init_op)

saver = tf.train.Saver(network_vars)

summary_writer = tf.summary.FileWriter(config.summary_path, graph=sess.graph)

"""create dictionary to be saved in json""" #talves fazer em outra funcao
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

if not os.path.exists(config.models_path):
  if config.restore:
    raise ValueError('There is no model to be restore from:' + config.models_path)
  else:
    os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)

if config.restore:
  if ckpt:
    print 'Restoring from ', ckpt.model_checkpoint_path
    saver.restore(sess,ckpt.model_checkpoint_path)
    tamanho=len(ckpt.model_checkpoint_path.split('-'))
    initialIteration = int(ckpt.model_checkpoint_path.split('-')[tamanho-1])
    
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
    initialIteration = 1
else:
    ckpt = 0
    initialIteration = 1

print 'Logging into ' + config.summary_path
"""Training"""
lowest_error = 150000
lowest_val  = 150000
lowest_iter = 1
lowest_val_iter = 1
"""training loop"""
training_start_time =time.time()
print config.n_epochs*dataset.getNImagesDataset()/config.batch_size
for i in range(initialIteration, config.n_epochs*dataset.getNImagesDataset()/config.batch_size):
  epoch_number = (float(i)*float(config.batch_size))/float(dataset.getNImagesDataset())
  """Save the model every model_saving_period iterations"""
  if i%config.model_saving_period == 0:
    saver.save(sess, config.models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'


  start_time = time.time()

  batch = dataset.train.next_batch(config.batch_size)
  feedDict={tf_images: batch[0], tf_points: batch[1],lr: (config.learning_rate/(config.lr_update_value ** int(int(epoch_number)/config.lr_update_period)))}
  sess.run(train_step, feed_dict=feedDict)

  duration = time.time() - start_time

  if i%8 == 0:
    examples_per_sec = config.batch_size / duration
    result=sess.run(loss_function, feed_dict=feedDict)
    result > 0
   # print sess.run(tf_points, feed_dict=feedDict)[0]
    #print sess.run(last_layer, feed_dict=feedDict)[0]
    #print(result)
    train_accuracy = sum(result)/config.batch_size
    if  train_accuracy < lowest_error:
      lowest_error = train_accuracy
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"
        %(epoch_number, i, i*config.batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))
  if i%config.summary_writing_period == 1 and (config.use_tensorboard or config.save_json_summary):
    output, result, sim_input = sess.run([last_layer,loss_function, x], feed_dict=feedDict)
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
  if i%config.validation_period == 0:
    validation_result_error = 0
    for j in range(0,dataset.getNImagesValidation()/(config.batch_size)):
      batch_val = dataset.validation.next_batch(config.batch_size)
      feedDictVal={tf_images: batch_val[0], tf_points: batch_val[1]}
      #print sess.run(tf_points, feed_dict=feedDictVal)[0]
      #print sess.run(last_layer, feed_dict=feedDictVal)[0]
      result = sess.run(loss_function, feed_dict=feedDictVal)
      validation_result_error += sum(result)
    if validation_result_error:
      validation_result_error = validation_result_error[0][0]
    if dataset.getNImagesValidation() !=0 :
      validation_result_error = (validation_result_error)/dataset.getNImagesValidation()
    if config.use_tensorboard:
      summary_str_val=sess.run(val_summary, feed_dict={val_error: validation_result_error})
      summary_writer.add_summary(summary_str_val,i)
    if config.save_json_summary:
      dados['variable_errors_val'].append(validation_result_error)
      outfile= open(config.models_path +'summary.json','w')
      json.dump(dados, outfile)
      outfile.close()