#visualizacao por inversao da rede
#le dados do config
#parametros: os indices das imagens do conjunto de validacao a serem usadas como entrada
"""Deep dive libs"""
from input_data_levelDB_simulator import DataSetManager
from config import *
from utils import *
from features_optimization import *

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
import glob
from optparse import OptionParser
from PIL import Image
import subprocess
import time
from ssim_tf import ssim_tf
from scipy import misc

fig_ind=0

def read_image(image_name):
	image = Image.open(image_name).convert('RGB')
	#image = image.resize((input_size[0], input_size[1]), Image.ANTIALIAS)
	image = np.asarray(image)
	image = image.astype(np.float32)
	image = np.multiply(image, 1.0 / 255.0)
	return image

def optimize_input_to_ground_truth(input_size, x, y,ground_truth):

 config= configOptimization()
 images = np.empty((1,)+input_size)
 #img_noise = np.random.uniform(low=0.0, high=1.0, size=input_size)
 img_noise = np.zeros(input_size) + 0.5
 #img_noise = ground_truth
 sess=tf.get_default_session()
 t_score = tf.reduce_mean(tf.abs(tf.subtract(y, ground_truth)))
 t_grad = tf.gradients(t_score, x)[0]

 if config.lap_grad_normalization:
  grad_norm=lap_normalize(t_grad[0,:,:,:])
 else:
  grad_norm=normalize_std(t_grad)

 images[0] = img_noise.copy()

 opt_error=[]
 for i in xrange(1,config.opt_n_iters+1):
	  feedDict={x: images}
	  g, score = sess.run([grad_norm, t_score], feed_dict=feedDict)
	  opt_error.append(score)
	  images[0] = images[0]-g*config.opt_step
	  #l2 decay
	  if config.decay:
	   images[0] = images[0]*(1-config.decay)
	  #gaussian blur
	  if config.blur_iter:
	   if i%config.blur_iter==0:
	    images[0] = gaussian_filter(images[0], sigma=config.blur_width)
	  #clip norm
	  if config.norm_pct_thrshld:
	   norms=np.linalg.norm(images[0], axis=2, keepdims=True)
	   n_thrshld=np.sort(norms, axis=None)[int(norms.size*config.norm_pct_thrshld)]
	   images[0]=images[0]*(norms>=n_thrshld)
	  # #clip contribution
	  if config.contrib_pct_thrshld:
	   contribs=np.sum(images[0]*g[0], axis=2, keepdims=True)
	   c_thrshld=np.sort(contribs, axis=None)[int(contribs.size*config.contrib_pct_thrshld)]
	   images[0]=images[0]*(contribs>=c_thrshld)

 #global fig_ind
 #global img_inds
 #error = np.array(opt_error)
 #plt.figure(img_inds[fig_ind])
 #plt.grid(True)
 #plt.suptitle("optmization error of " + str(img_inds[fig_ind]))
 #axes = plt.gca()
 #fig_ind+=1
 #plt.plot(range(0,len(opt_error)), error)
 #plt.show(block=False)

 return images[0].astype(np.float32), score

"""Verifying options integrity"""
config = configVisualization()

if config.save_features_to_disk not in (True, False):
  raise Exception('Wrong save_features_to_disk option. (True or False)')
if config.save_json_summary not in (True, False):
  raise Exception('Wrong save_json_summary option. (True or False)')
if config.use_tensorboard not in (True, False):
  raise Exception('Wrong use_tensorboard option. (True or False)')

dataset = DataSetManager(config)
global_step = tf.Variable(0, trainable=False, name="global_step")

""" Creating section"""
x = tf.placeholder("float", shape= (None,)+config.input_size , name="input_image")
y_ = tf.placeholder("float", name="output_image")
sess = tf.InteractiveSession()
with tf.variable_scope("network", reuse=None):
  last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout,training=False)

saver = tf.train.Saver(tf.global_variables())

init_op=tf.global_variables_initializer()
sess.run(init_op)
summary_writer =  tf.summary.FileWriter(config.summary_path, graph=sess.graph)

"""Load a previous model if restore is set to True"""

if not os.path.exists(config.models_path):
  os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)

if ckpt:
 print 'Restoring from ', ckpt.model_checkpoint_path  
 saver.restore(sess,ckpt.model_checkpoint_path)  
else:
  print 'Can\'t Restore from ', config.models_path
  sys.exit()

print 'Logging into ' + config.summary_path

ft_ops=[]
weights=[]
for key in config.features_list:
  ft_ops.append(feature_maps[key][0])
  weights.append(feature_maps[key][1]) 

path=config.summary_path
loss_function=tf.reduce_mean(tf.abs(tf.subtract(last_layer, y_)))

img_inds=map(int, sys.argv[1:])

""" Optimization """
print("Running Optimization")
for i in xrange(dataset.getNImagesValidation()):
  ground_truth=(dataset.validation.next_batch(1)[0])[0]
  if i in img_inds:
        opt,score=optimize_input_to_ground_truth(ground_truth.shape, x, last_layer, ground_truth)
	feedDict=({x: np.expand_dims(opt,0), y_: ground_truth})
	output, error = sess.run([last_layer,loss_function], feed_dict=feedDict)
	ft_maps=sess.run(ft_ops, feed_dict=feedDict)
	#print i, error
        if config.use_tensorboard:
          opt_summary=tf.summary.image("inversion", np.expand_dims(opt,0))
	  gt_summary=tf.summary.image("inversion ground truth", np.expand_dims(ground_truth.astype(np.float32),0))
	  out_summary=tf.summary.image("inversion output", output)
          summary_str_opt,summary_str_gt,summary_str_out=sess.run([opt_summary, gt_summary, out_summary])
          summary_writer.add_summary(summary_str_opt,i)
	  summary_writer.add_summary(summary_str_gt,i)
	  summary_writer.add_summary(summary_str_out,i)
        # salvando as imagens como bmp
        if(config.save_features_to_disk):
	  opt-=opt.min()
	  opt/=max(opt.max(),0.0000001)
          opt_img=(opt * 255).round().astype(np.uint8)
          im = Image.fromarray(opt_img)
          file_name="opt.bmp"
          folder_name=path+"/opt/"+str(i)
          if not os.path.exists(folder_name):
          	os.makedirs(folder_name)
          im.save(folder_name+"/"+file_name)

	  ground_truth-=ground_truth.min()
	  ground_truth/=max(ground_truth.max(),0.0000001)
          gt_img=(ground_truth * 255).round().astype(np.uint8)
          im = Image.fromarray(gt_img)
          file_name="gt.bmp"
          folder_name=path+"/opt/"+str(i)
          if not os.path.exists(folder_name):
          	os.makedirs(folder_name)
          im.save(folder_name+"/"+file_name)

	  output[0]-=output[0].min()
	  output[0]/=max(output[0].max(),0.0000001)
          out_img=(output[0] * 255).round().astype(np.uint8)
          im = Image.fromarray(out_img)
          file_name="output.bmp"
          folder_name=path+"/opt/"+str(i)
          if not os.path.exists(folder_name):
          	os.makedirs(folder_name)
          im.save(folder_name+"/"+file_name)

        for ft, key in zip(ft_maps, config.features_list):
           ft_grid=put_features_on_grid_np(ft)
           ft_name="Features_map_"+key
           ft_grid_img=(ft_grid-ft_grid.min())
           ft_grid_img*=(255/ft_grid_img.max())
           ft_grid_img=ft_grid_img.astype(np.uint8)
           ft_grid_im = Image.fromarray(ft_grid_img[0,:,:,0])
           ft_grid_folder_name=config.summary_path
           if not os.path.exists(ft_grid_folder_name):
             os.makedirs(ft_grid_folder_name)
           ft_grid_im.save(ft_grid_folder_name+"/opt/"+str(i)+"/"+ft_name+".png")


plt.show()
