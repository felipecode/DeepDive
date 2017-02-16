"""Deep dive libs"""
from input_data_levelDB_simulator import DataSetManager
from config import *
from utils import *
from features_optimization import optimize_feature
from simulator import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from resnet_12 import create_structure
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
from scipy import misc

import json

"""Verifying options integrity"""
def verifyConfig(config):
  if config.save_features_to_disk not in (True, False):
    raise Exception('Wrong save_features_to_disk option. (True or False)')
  if config.save_json_summary not in (True, False):
    raise Exception('Wrong save_json_summary option. (True or False)') 
  if config.use_tensorboard not in (True, False):
    raise Exception('Wrong use_tensorboard option. (True or False)')
  if config.save_error_transmission not in (True, False):
    raise Exception('Wrong save_error_transmission option. (True or False)')
  if config.use_deconv not in (True, False):
    raise Exception('Wrong use_deconv option. (True or False)')
  if config.use_depths not in (True, False):
    raise Exception('Wrong use_depths option. (True or False)')

"""Verifying options integrity"""
config = configVisualization()
verifyConfig(config)

""" Creating section"""
sess = tf.InteractiveSession()
c,binf,range_array=acquireProperties(config,sess)
dataset = DataSetManager(config) 
global_step = tf.Variable(0, trainable=False, name="global_step")

"""creating plaholders"""
batch_size=config.batch_size
tf_images=tf.placeholder("float",(None,) +config.input_size, name="images")
tf_depths=tf.placeholder("float",(None,) +config.depth_size, name="depths")
tf_range=tf.placeholder("float",(None,)+range_array.shape[1:], name="ranges")
tf_c=tf.placeholder("float",(None,)+c.shape[1:], name="c")
tf_binf=tf.placeholder("float",(None,)+binf.shape[1:], name="binf")
lr = tf.placeholder("float", name = "learning_rate")

"""defining simulator structure"""
y_image = tf_images
x=applyTurbidity(y_image, tf_depths, tf_c, tf_binf, tf_range)


with tf.variable_scope("network", reuse=None):
  last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout,training=False)

" Creating comparation metrics"
mse_loss = tf.reduce_mean(tf.abs(tf.sub(255.0*last_layer, 255.0*y_image)), reduction_indices=[1,2,3])
loss_function = mse_loss

"""Creating summaries"""

tf.summary.image('Input', x)
tf.summary.image('Output', last_layer)
tf.summary.image('GroundTruth', y_image)

ft_ops=[]
weights=[]
for key in config.features_list:
  ft_ops.append(feature_maps[key][0])
  weights.append(feature_maps[key][1])  
for key in scalars:
  tf.summary.scalar(key,scalars[key])
for key in config.histograms_list:
 tf.histogram_summary('histograms_'+key, histograms[key])
tf.summary.scalar('Loss', tf.reduce_mean(loss_function))

summary_op = tf.summary.merge_all()
saver = tf.train.Saver(tf.global_variables())

init_op=tf.global_variables_initializer()
sess.run(init_op)
summary_writer = tf.summary.FileWriter(config.summary_path, graph=sess.graph)

if not os.path.exists(config.models_path):
  os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)

dados={}
dados['summary_writing_period']=config.summary_writing_period
dados['batch_size']=config.batch_size
dados['variable_errors']=[]
dados['time']=[]
dados['variable_errors_val']=[]

if ckpt:
 print 'Restoring from ', ckpt.model_checkpoint_path  
 saver.restore(sess,ckpt.model_checkpoint_path)
 if config.save_json_summary:
    if os.path.isfile(config.summary_path +'summary.json'):
      outfile= open(config.summary_path +'summary.json','r+')
      dados=json.load(outfile)
      outfile.close()
    else:
      outfile= open(config.summary_path +'summary.json','w')
      json.dump(dados, outfile)
      outfile.close()   
else:
  print 'Can\'t Restore from ', config.models_path
  sys.exit()

print 'Logging into ' + config.summary_path

initialIteration = 1

training_start_time =time.time()

max_actvs=[]
for key in config.features_list:
  "descobrindo o tamanho de cada feature map"
  ft_shape=feature_maps[key][0].get_shape()
  "inicializando a variavel da ativacao maxima"
  init_img=np.zeros((config.num_top_actvs,)+config.input_size+(ft_shape[3],),dtype=np.float32)-float("inf") 
  init_actv=np.zeros((config.num_top_actvs,)+tuple(ft_shape[1:]),dtype=np.float32)-float("inf")
  init_avg=np.zeros((config.num_top_actvs,)+(ft_shape[3],),dtype=np.float32)-float("inf")
  max_actvs.append((init_img,init_actv,init_avg))
  dados[key]=[]

""" Optimization """
print("Running Optimization")
for key, channel in config.features_opt_list:	
        ft=feature_maps[key][0]
        n_channels=ft.get_shape()[3]
	opt_grid=np.empty((1,)+config.input_size+(n_channels,))
        if channel<0:
          #otimiza todos os canais      
          for ch in xrange(n_channels):
            opt_output=optimize_feature(config.input_size, x, ft[:,:,:,ch])
            #if config.use_tensorboard:
            #  opt_name="optimization_"+key+"_"+str(ch).zfill(len(str(n_channels)))
            #  opt_summary=tf.summary.image(opt_name, np.expand_dims(opt_output,0))
            #  summary_str=sess.run(opt_summary)
            #  summary_writer.add_summary(summary_str,0)
          # salvando as imagens como bmp
            if(config.save_features_to_disk):
              save_optimized_image_to_disk(opt_output,ch,n_channels,key,config.summary_path)
            opt_output -= opt_output.min()
            opt_output *= (255/(opt_output.max()+0.0001))
            opt_grid[0,:,:,:,ch]=opt_output
          #opt_grid_img=put_grads_on_grid_np(opt_grid.astype(np.float32))
	  if config.use_tensorboard:
           opt_grid_name="opt_"+key
	   opt_grid_img=put_grads_on_grid_np(opt_grid.astype(np.float32))
           opt_grid_summary=tf.summary.image(opt_grid_name, opt_grid_img)
           opt_grid_summary_str=sess.run(opt_grid_summary)
           summary_writer.add_summary(opt_grid_summary_str,0)
          #opt_grid_img=(opt_grid_img-opt_grid_img.min())
          #opt_grid_img*=(255/opt_grid_img.max())
          #opt_grid_img=opt_grid_img.astype(np.uint8)
          #opt_im = Image.fromarray(opt_grid_img[0,:,:,:])
          #folder_name=config.summary_path
          #if not os.path.exists(folder_name):
          #  os.makedirs(folder_name)
          #opt_im.save(folder_name+"/"+opt_grid_name+".bmp")
	    
        else:
          opt_output=optimize_feature(config.input_size, x, ft[:,:,:,channel])
          if config.use_tensorboard:
            opt_name="optimization_"+key+"_"+str(channel).zfill(len(str(n_channels)))
            opt_summary=tf.summary.image(opt_name, np.expand_dims(opt_output,0))
            summary_str=sess.run(opt_summary)
            summary_writer.add_summary(summary_str,0)
          # salvando as imagens como bmp
          if(config.save_features_to_disk):
            save_optimized_image_to_disk(opt_output,channel,n_channels,key,config.summary_path)

print("Images in dataset: %d"%(dataset.getNImagesDataset()))
for i in range(initialIteration, dataset.getNImagesDataset()/config.batch_size+1):
  epoch_number = 1.0 + (float(i)*float(config.batch_size))/float(dataset.getNImagesDataset())
  start_time = time.time()

  batch = dataset.train.next_batch(config.batch_size)
  if config.use_depths:
    feedDict={tf_images: batch[0], tf_depths: batch[1], tf_range: range_array, tf_c: c, tf_binf: binf}
  else:
    constant_depths=np.ones((batch_size,)+config.depth_size, dtype=np.float32);
    depths=constant_depths*10*np.random.rand(batch_size,1,1,1)
    feedDict={tf_images: batch[0], tf_depths: depths, tf_range: range_array, tf_c: c, tf_binf: binf}

  sim_output=sess.run(x,feed_dict=feedDict)
  if len(ft_ops) > 0:
      ft_maps= sess.run(ft_ops, feed_dict=feedDict)
  else:
      ft_maps= []

  for ft, actv, key in zip(ft_maps, max_actvs, config.features_list):
	batch_actv_sum=np.zeros(ft.shape[3],dtype=np.float32)
	"Percorre todo o batch"
	for j in xrange(ft.shape[0]):
		"percorre os canais do feature map"
		for k in xrange(ft.shape[3]):
			ft_avg=np.average(ft[j,:,:,k])
			ft_pos=-1
			for p in xrange(0,config.num_top_actvs):
				if ft_avg>actv[2][p,k]:
					ft_pos=p
					break
			if ft_pos>=0:
				for pos in xrange(config.num_top_actvs-1, ft_pos, -1):
					actv[0][pos,:,:,:,k]=actv[0][pos-1,:,:,:,k]
					actv[1][pos,:,:,k]=actv[1][pos-1,:,:,k]
					actv[2][pos,k]=actv[2][pos-1,k]
				actv[0][ft_pos,:,:,:,k]=sim_output[j,:,:,:]
				actv[1][ft_pos,:,:,k]=ft[j,:,:,k]
				actv[2][ft_pos,k]=ft_avg
  	dados[key].append(ft.mean(axis=(0,1,2)).tolist())
	
  duration = time.time() - start_time

  if i%4 == 0:
    examples_per_sec = config.batch_size / duration
    print("step %d, images used %d, examples per second %f"
        %(i, i*config.batch_size, examples_per_sec))

  if (i%config.summary_writing_period == 1 or i==(dataset.getNImagesDataset()/config.batch_size)) and (config.use_tensorboard or config.save_features_to_disk or config.save_json_summary):
    output, result = sess.run([last_layer,loss_function], feed_dict=feedDict)
    result = np.mean(result)

    if config.use_deconv:
      deconv=deconvolution(x, feedDict, ft_ops, config.features_list, config.batch_size, config.input_size)
    else:
      deconv=[None]*len(ft_ops)

    if config.save_json_summary:
      dados['variable_errors'].append(float(result))
      dados['time'].append(time.time() - training_start_time)
      outfile = open(config.summary_path +'summary.json','w')
      json.dump(dados, outfile)
      outfile.close()
    if config.use_tensorboard:
      summary_str = sess.run(summary_op, feed_dict=feedDict)
      summary_writer.add_summary(summary_str,i)
      if len(ft_ops) > 0:
        for ft, w, d, actv, key in zip(ft_maps, weights, deconv, max_actvs, config.features_list):
          for ch in xrange(ft.shape[3]):
		summary_name=key+"_"+str(ch).zfill(len(str(ft.shape[3])))
          	avg_actv_summary=tf.summary.scalar(summary_name, np.average(ft[:,:,:,ch]))
          	avg_actv_str = sess.run(avg_actv_summary, feed_dict=feedDict)
          	summary_writer.add_summary(avg_actv_str,i) 		
          ft_grid=put_features_on_grid_np(ft)
          ft_name="Features_map_"+key
          ft_summary=tf.summary.image(ft_name, ft_grid)
          summary_str=sess.run(ft_summary)
          summary_writer.add_summary(summary_str,i)
          if w is not None:
            kernel=w.eval()
            kernel_grid=put_kernels_on_grid_np(kernel)
            kernel_name="kernels_"+key
            kernel_summary=tf.summary.image(kernel_name, kernel_grid)
            kernel_summary_str=sess.run(kernel_summary)
            summary_writer.add_summary(kernel_summary_str,i)
          if d is not None:
            deconv_grid=put_grads_on_grid_np(d.astype(np.float32))
            deconv_name="deconv_"+key
            deconv_summary=tf.summary.image(deconv_name, deconv_grid)
            deconv_summary_str=sess.run(deconv_summary)
            summary_writer.add_summary(deconv_summary_str,i)
#          max_actv_grid=put_features_on_grid_np(np.expand_dims(actv[1][:,:,0],0))
#          max_actv_name="max_actv_"+key
#          max_actv_summary=tf.summary.image(max_actv_name, max_actv_grid)
#          max_actv_summary_str=sess.run(max_actv_summary)
#          summary_writer.add_summary(max_actv_summary_str,i)
#          max_actv_input_grid=put_grads_on_grid_np(np.expand_dims(actv[0][:,:,:,0],0))
#          max_actv_input_name="max_actv_inputs_"+key
#          max_actv_input_summary=tf.summary.image(max_actv_input_name, max_actv_input_grid)
#          max_actv_input_summary_str=sess.run(max_actv_input_summary)
#          summary_writer.add_summary(max_actv_input_summary_str,i)

    if(config.save_features_to_disk):
      save_images_to_disk(output,sim_output,batch[0],config.summary_path)
      save_feature_maps_to_disk(ft_maps, weights, deconv, config.features_list,config.summary_path)

for actv, key in zip(max_actvs, config.features_list):
 if(config.save_features_to_disk):
   save_max_activations_to_disk(max_actvs, config.features_list,config.summary_path)

   for n in xrange(0,config.num_top_actvs):         
     max_actv_grid=put_features_on_grid_np(np.expand_dims(actv[1][n,:,:,:],0))
     max_actv_name="max_actv_"+key+"_N_"+str(n).zfill(len(str(config.num_top_actvs)))         
     max_actv_img=(max_actv_grid-max_actv_grid.min())
     max_actv_img*=(255/(max_actv_img.max()+0.0001))
     max_actv_img=max_actv_img.astype(np.uint8)
     max_im = Image.fromarray(max_actv_img[0,:,:,0])
     max_folder_name=config.summary_path
     if not os.path.exists(max_folder_name):
	os.makedirs(max_folder_name)
     max_im.save(max_folder_name+"/"+max_actv_name+".png")
     
     max_actv_input_grid=put_grads_on_grid_np(np.expand_dims(actv[0][n,:,:,:,:],0))
     max_actv_input_name="max_actv_inputs_"+key+"_N_"+str(n).zfill(len(str(config.num_top_actvs)))
     max_actv_input_img=(max_actv_input_grid-max_actv_input_grid.min())
     max_actv_input_img*=(255/(max_actv_input_img.max()+0.0001))
     max_actv_input_img=max_actv_input_img.astype(np.uint8)
     max_in_im = Image.fromarray(max_actv_input_img[0,:,:,:])
     max_folder_name=config.summary_path
     if not os.path.exists(max_folder_name):
	os.makedirs(max_folder_name)
     max_in_im.save(max_folder_name+"/"+max_actv_input_name+".png")

 if(config.use_tensorboard):
     max_actv_grid=put_features_on_grid_np(actv[1])
     max_actv_name="max_actv_"+key
     max_actv_summary=tf.summary.image(max_actv_name, max_actv_grid,max_outputs=config.num_top_actvs)
     max_actv_summary_str=sess.run(max_actv_summary)
     summary_writer.add_summary(max_actv_summary_str,i)

     max_actv_input_grid=put_grads_on_grid_np(actv[0])
     max_actv_input_name="max_actv_inputs_"+key
     max_actv_input_summary=tf.summary.image(max_actv_input_name, max_actv_input_grid,max_outputs=config.num_top_actvs)
     max_actv_input_summary_str=sess.run(max_actv_input_summary)
     summary_writer.add_summary(max_actv_input_summary_str,i)

