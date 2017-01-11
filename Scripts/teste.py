"""Deep dive libs"""
from input_data_dive_test import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
from underwater_pathfinder import create_structure

"""Core libs"""
#import tensorflow as tf
import numpy as np
import scipy.signal as sig
from numpy import unravel_index

"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""Python libs"""
import os
from optparse import OptionParser
from PIL import Image
import Image, ImageDraw
import subprocess
import time
import glob
h_kernel = 128
w_kernel = 128
config= configDehazenet()
im_names =  glob.glob(config.evaluate_out_path + "*.png")
for name in im_names:
  #print name
  im = Image.open(name).convert('L')
  img = np.array(im,dtype=np.int64)
  #print np.sum(img)
  start=time.time()
  for i in xrange(1,img.shape[0]):
    img[i,0]=img[i,0]+img[i-1,0]

  for j  in xrange(1,img.shape[1]):
    img[0,j]=img[0,j]+img[0,j-1]

  for i in xrange(1,img.shape[0]):
    for j in xrange(1,img.shape[1]):
      img[i,j]=img[i,j]+img[i-1,j]+img[i,j-1]-img[i-1,j-1]  #OK

  #print img[img.shape[0]-1,img.shape[1]-1]


  minimo = 9999999999999999999
  indice_i=-1
  indice_j=-1 
  for i in xrange(img.shape[0]-h_kernel):
    for j in xrange(img.shape[1]-w_kernel):
      temp = img[i+h_kernel,j+w_kernel]+img[i,j]-(img[i+h_kernel,j]+img[i,j+w_kernel])
      if temp < minimo:
        minimo = temp
        indice_i=i
        indice_j=j

#  print minimo
#  print indice_i
#  print indice_j
  print time.time()-start 
  ponto = (indice_i,indice_j)
  #print img.shape
  #print ponto
  resultado = Image.open(name.replace("antigo/","")).convert('RGB')
  dr = ImageDraw.Draw(resultado)
  ################ Rectangle ###################
  cor = (ponto[1],ponto[0], ponto[1]+w_kernel,ponto[0]+h_kernel) # (x1,y1, x2,y2)
  dr.rectangle(cor, outline="red")
  ###############                ####################
  resultado.save(name.replace(".png", "_final.png"))


 
