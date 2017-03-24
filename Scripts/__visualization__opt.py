#script de visualizacao por maximizacao da ativacao
#otimiza todos os canais de um feature map em um determinado intervalo
#necassario para feature maps com muitos canais
#porque o codigo tem vazamento de memoria, e fica muito lento se rodar por muito tempo
#le parametros do arquivo de config
#salva os resultados parciais em um arquivo temporario
#le o arquivo temporario a cada execucao
#quando chega no ultimo canal, apaga o arquivo temporario e salvo os resultados na pasta e/ou manda pro tensorboard
#recebe 3 parametros: chave do feature map, canal de inicio, canal de fim
#exemplo: para visualizar um feature map chamado conv5 com 1024 canais, rode o seguinte script:
#python __visualization__opt conv5 0 100
#python __visualization__opt conv5 100 200
#python __visualization__opt conv5 200 300
#                   ...
#python __visualization__opt conv5 900 1000
#python __visualization__opt conv5 1000 1100
#nao faz mal se passar do numero de canais do feature map, o programa para automaticamente
"""Deep dive libs"""
from input_data_levelDB_simulator import DataSetManager
from config import *
from utils import *
from features_optimization import optimize_feature

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

"""creating plaholders"""
x = tf.placeholder("float", (None,)+config.input_size, name="input_image")

with tf.variable_scope("network", reuse=None):
  last_layer, dropoutDict, feature_maps,scalars,histograms = create_structure(tf, x,config.input_size,config.dropout,training=False)

saver = tf.train.Saver(tf.global_variables())

init_op=tf.global_variables_initializer()
sess.run(init_op)
summary_writer = tf.summary.FileWriter(config.summary_path, graph=sess.graph)

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

key=sys.argv[1]
start_channel=int(sys.argv[2])
end_channel=int(sys.argv[3])

""" Optimization """	
ft=feature_maps[key][0]
n_channels=ft.get_shape()[3]
opt_grid=np.empty((1,)+config.input_size+(n_channels,))
if os.path.exists(key+"_opt.npy"):
	print "Loading Data"
	opt_grid=np.load(key+"_opt.npy")
	print "Loading Done"

#otimiza todos os canais
print("Running Optimization")     
for ch in xrange(start_channel, min(n_channels, end_channel)):
    current_time=time.asctime(time.localtime(time.time()))
    print "optmizing "+key+" channel "+str(ch), current_time
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

if(ch>=(n_channels-1)):
  if config.use_tensorboard:    
    opt_grid_name="opt_"+key
    opt_grid_img=put_grads_on_grid_np(opt_grid.astype(np.float32))
    opt_grid_summary=tf.summary.image(opt_grid_name, opt_grid_img)
    opt_grid_summary_str=sess.run(opt_grid_summary)
    summary_writer.add_summary(opt_grid_summary_str,0)
  if os.path.exists(key+"_opt.npy"):
    os.remove(key+"_opt.npy")
  #salva uma imagem com todos os resultados em forma de grid
  #opt_grid_name="opt_"+key
  #opt_grid_img=put_grads_on_grid_np(opt_grid.astype(np.float32))
  #opt_grid_img=(opt_grid_img-opt_grid_img.min())
  #opt_grid_img*=(255/opt_grid_img.max())
  #opt_grid_img=opt_grid_img.astype(np.uint8)
  #opt_im = Image.fromarray(opt_grid_img[0,:,:,:])
  #folder_name=config.summary_path
  #if not os.path.exists(folder_name):
  #  os.makedirs(folder_name)
  #opt_im.save(folder_name+"/"+opt_grid_name+".bmp")
else:
  print "Saving Data"
  np.save(key+"_opt.npy", opt_grid)
  print "Done"
