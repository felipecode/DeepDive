"""Deep dive libs"""
from input_data_levelDB_simulator_data_augmentation import DataSetManager
from config import *
from utils import *
from features_optimization import optimize_feature

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from loss_network import *
from simulator import *
from inception_res_BACBAC_normalized import create_structure
from discriminator_net_test import create_discriminator_structure

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
import glob

import json

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
  if config.save_error_transmission not in (True, False):
    raise Exception('Wrong save_error_transmission option. (True or False)')
  if config.use_deconv not in (True, False):
    raise Exception('Wrong use_deconv option. (True or False)')
  if config.use_depths not in (True, False):
    raise Exception('Wrong use_depths option. (True or False)')



config = configMainSimulator()
verifyConfig(config)
""" Creating section"""
sess = tf.InteractiveSession()
c,binf,range_array=acquireProperties(config,sess)
dataset = DataSetManager(config) 
print dataset.getNImagesDataset()
global_step = tf.Variable(0, trainable=False, name="global_step")

"""creating plaholders"""
batch_size=config.batch_size
tf_images=tf.placeholder("float",(batch_size,) +config.input_size, name="images")  #groundtruth
tf_depths=tf.placeholder("float",(batch_size,) +config.depth_size, name="depths")  #depth
tf_range=tf.placeholder("float",range_array.shape, name="ranges")   # multiplicative gain to depth
tf_c=tf.placeholder("float",c.shape, name="c")          # scattering constant
tf_binf=tf.placeholder("float",binf.shape, name="binf")   
lr = tf.placeholder("float", name = "learning_rate")

"""defining simulator structure"""
y_image = tf_images
x=applyTurbidity(y_image, tf_depths, tf_c, tf_binf, tf_range)

"""defining generative network structure"""
with tf.variable_scope("network", reuse=None):
  last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout)

#network_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')#pega as variaveis da rede
network_vars=tf.get_collection(tf.GraphKeys.VARIABLES, scope='network')
#se der erro aqui e porque ta com uma versao antiga do tensorflow, ai tem que usar tf.GraphKeys.VARIABLES

"""define structure feature loss"""
feature_loss=create_loss_structure(tf, 255.0*last_layer, 255.0*tf_images, sess)
if config.use_discriminator:
  """define discriminator structure for both ground truth and output"""
  with tf.variable_scope('discriminator', reuse=None):
    d_score_gt=create_discriminator_structure(tf,y_image,config.input_size)
  with tf.variable_scope('discriminator', reuse=True):
    d_score_output=create_discriminator_structure(tf,last_layer,config.input_size)
  discriminator_vars=tf.get_collection(tf.GraphKeys.VARIABLES, scope='discriminator')

""" Creating losses for generative network"""
#lab_mse_loss = tf.reduce_mean(np.absolute(np.subtract(color.rgb2lab((255.0*last_layer).eval()), color.rgb2lab((255.0*y_image).eval()))))
mse_loss = tf.reduce_mean(tf.abs(tf.sub(255.0*last_layer, 255.0*y_image)), reduction_indices=[1,2,3])


if config.use_discriminator:
  discriminator_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_score_output, tf.ones_like(d_score_output)))#tf.reduce_mean(-tf.log(tf.clip_by_value(tf.nn.sigmoid(d_score_output),1e-10,1.0)))#com log puro tava dando log(0)=NaN depois de um tempo

if config.use_discriminator:
  loss_function = (feature_loss+1000*discriminator_loss)/2
else:
  loss_function = feature_loss

if config.use_discriminator:
  """ Loss for descriminative network"""

  cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(d_score_gt, tf.ones_like(d_score_gt))
  disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
      
  cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(d_score_output, tf.zeros_like(d_score_output))

  disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')

  discriminator_error = tf.add(disc_real_loss, disc_fake_loss)
  #discriminator_error=tf.reduce_mean(-(tf.log(tf.clip_by_value(d_score_gt,1e-10,1.0))+tf.log(tf.clip_by_value(1-d_score_output,1e-10,1.0))))

  disc_train_step = tf.train.AdamOptimizer(learning_rate = lr, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon,
                                    use_locking=config.use_locking).minimize(discriminator_error, var_list=discriminator_vars)#treina so o discriminador, sem mexer nos pesos da rede

train_step = tf.train.AdamOptimizer(learning_rate = lr, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon,
                                    use_locking=config.use_locking).minimize(loss_function, var_list=network_vars)#treina so a rede, nao o discriminador


"""Creating summaries"""

tf.image_summary('Input', x)
tf.image_summary('Output', last_layer)
tf.image_summary('GroundTruth', y_image)

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
if config.use_discriminator:
  tf.scalar_summary('discriminator_score_groundtruth', tf.nn.sigmoid(tf.reduce_mean(d_score_gt)))
  tf.scalar_summary('discriminator_score_output', tf.nn.sigmoid(tf.reduce_mean(d_score_output)))
  tf.scalar_summary('discriminator_loss', discriminator_loss)
  tf.scalar_summary('discriminator_error', discriminator_error)

summary_op = tf.merge_all_summaries()

saver = tf.train.Saver(network_vars)
if config.use_discriminator:
  saver_discriminator = tf.train.Saver(discriminator_vars)

init_op=tf.initialize_all_variables()
sess.run(init_op)

summary_writer = tf.train.SummaryWriter(config.summary_path, graph=sess.graph)

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

"""Load a previous model if restore is set to True"""

if not os.path.exists(config.models_path):
  if config.restore:
    raise ValueError('There is no model to be restore from:' + config.models_path)
  else:
    os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)

if config.use_discriminator:
  if not os.path.exists(config.models_discriminator_path):
    if config.restore_discriminator:
      raise ValueError('There is no model to be restore from:' + config.models_discriminator_path)
    else:
      os.mkdir(config.models_discriminator_path)
  ckpt_discriminator = tf.train.get_checkpoint_state(config.models_discriminator_path)


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


if config.restore_discriminator:
  if ckpt_discriminator:
    print 'Restoring from ', ckpt_discriminator.model_checkpoint_path
    saver_discriminator.restore(sess,ckpt_discriminator.model_checkpoint_path)

print 'Logging into ' + config.summary_path

"""Training"""
lowest_error = 150000;
lowest_val  = 150000;
lowest_iter = 1;
lowest_val_iter = 1;

"""training loop"""
training_start_time =time.time()
print config.n_epochs*dataset.getNImagesDataset()/config.batch_size
for i in range(initialIteration, config.n_epochs*dataset.getNImagesDataset()/config.batch_size):
  epoch_number = (float(i)*float(config.batch_size))/float(dataset.getNImagesDataset())
  """Save the model every model_saving_period iterations"""
  if i%config.model_saving_period == 0:
    saver.save(sess, config.models_path + 'model.ckpt', global_step=i)
    if config.use_discriminator:
      saver_discriminator.save(sess, config.models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'


  start_time = time.time()

  batch = dataset.train.next_batch(config.batch_size)
  if config.use_depths:
    feedDict={tf_images: batch[0], tf_depths: batch[1], tf_range: range_array, tf_c: c, tf_binf: binf, lr: (config.learning_rate/(config.lr_update_value ** int(int(epoch_number)/config.lr_update_period)))}
  else:
    constant_depths=np.ones((batch_size,)+config.depth_size, dtype=np.float32);
    depths=constant_depths*10*np.random.rand(batch_size,1,1,1)
    feedDict={tf_images: batch[0], tf_depths: depths, tf_range: range_array, tf_c: c, tf_binf: binf, lr: (config.learning_rate/(config.lr_update_value ** int(int(epoch_number)/config.lr_update_period)))}
  if config.use_discriminator:
    sess.run([train_step,disc_train_step], feed_dict=feedDict)
  else:
    sess.run(train_step, feed_dict=feedDict)

  duration = time.time() - start_time

  if i%8 == 0:
    examples_per_sec = config.batch_size / duration
    result=sess.run(loss_function, feed_dict=feedDict)#,discriminator_error], feed_dict=feedDict)
    result > 0
    train_accuracy = sum(result)/config.batch_size
    if  train_accuracy < lowest_error:
      lowest_error = train_accuracy
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"
        %(epoch_number, i, i*config.batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))

  if i%config.summary_writing_period == 1 and (config.use_tensorboard or config.save_features_to_disk or config.save_json_summary):
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
    if(config.save_features_to_disk):
      save_images_to_disk(output,sim_input,batch[0],config.summary_path)
      save_feature_maps_to_disk(ft_maps, weights, deconv, config.features_list,config.summary_path)

  if i%config.validation_period == 0:
    validation_result_error = 0
    for j in range(0,dataset.getNImagesValidation()/(config.batch_size)):
      #batch_val = dataset.validation.next_batch(config.batch_size)
      batch_val = dataset.validation.next_batch(config.batch_size)
      feedDictVal={tf_images: batch_val[0], tf_depths: batch_val[1], tf_range: range_array, tf_c: c, tf_binf: binf}
      #turbid_images=applyTurbidity(tf_images, tf_depths, tf_c, tf_binf, tf_range)
      #result=sess.run(turbid_images, feed_dict=feedDict)
      #feedDictVal = {x: result, y_: batch_val[0]}
      result = sess.run(loss_function, feed_dict=feedDictVal)
      validation_result_error += sum(result)
      if config.save_error_transmission:
        error_per_transmission=[0.0] * config.num_bins
        count_per_transmission=[0] * config.num_bins
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
