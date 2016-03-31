

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
import gzip
import os
import numpy as np
import Image, colorsys
from scipy import misc
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import glob

import matplotlib.pyplot as plt

from config import *

#so pra testar mesmo
#n_images_dataset = 10000

class DataSetConverter(object):

  def __init__(self, path, path_val, input_size, proportions, n_images_dataset):
    self.input_size = input_size
    self.im_names = []
    self.im_names_val = []
    self.tracker = 0
    self.tracker_val = 0
    #n_images_dataset_val = int(n_images_dataset*0.1)

    """ First thing we do is to elect a folder number based on the proportions vec """
    #proportions
    
    #n_images_fold = int(proportions[i]*n_images_dataset)
    #n_images_fold_val = int(proportions[i]*n_images_dataset_val)

    #path_folder = path + '/' + str(i+1)
    #path_folder_val = path_val + '/' + str(i+1)

    """Replace possible // for /"""
    #repls = ('//', '/'), ('', '')
    #path_folder = reduce(lambda a, kv: a.replace(*kv), repls, path_folder)
    #path_folder_val = reduce(lambda a, kv: a.replace(*kv), repls, path_folder)

    self.im_names  = glob.glob(path + "/*.jpg")
    self.im_names_val = glob.glob(path_val + "/*.jpg")

    #random.shuffle(im_names_fold)
    #im_names_fold = im_names_fold[0:n_images_fold]
    #im_names_fold_val = im_names_fold_val[0:n_images_fold_val]

    #self.im_names = self.im_names + im_names_fold
    #self.im_names_val = self.im_names_val + im_names_fold_val
      
    random.shuffle(self.im_names)
    #random.shuffle(self.im_names_val)


  def extract_dataset(self, n_images):
    """Creates chunkes of inputs and its respective labels and returns them into two 4D uint8 numpy array [index, y, x, depth]."""
    print 'Loading images...'
    images = []
    labels = []

    im_generated = 0
    
    while im_generated < n_images:
      if self.tracker >= len(self.im_names):
        self.tracker = 0
        random.shuffle(self.im_names)
      
      name = self.im_names[self.tracker]
      #print 'Opening image: ', name

      im = Image.open(name)
      im = im.resize((500, 500,3), PIL.Image.ANTIALIAS)

      repls = ('Training', 'GroundTruth'), ('', '')
      name = reduce(lambda a, kv: a.replace(*kv), repls, name)
        
      lb = Image.open(name) 

      lb = lb.resize((500, 500,3), PIL.Image.ANTIALIAS)

      im = np.asarray(im)
      lb = np.asarray(lb)

      assert im.size == lb.size

      height, width = im.shape[0], im.shape[1]

      """Generate chunks"""
      # for h in range(0, height-self.input_size[0], self.input_size[0]):
      #   for w in range(0, width-self.input_size[1], self.input_size[1]):
      #     h_end = h + self.input_size[0]
      #     w_end = w + self.input_size[1]

      #     chunk = im[h:h_end, w:w_end]
      #     images.append(chunk)

      #     chunk = lb[h: h_end, w:w_end]
      #     labels.append(chunk)

      #     im_generated += 1

      self.tracker += 1
#################################################################
#acho que e assim e mais rapido, tem que testar se funciona mesmo
#################################################################
    return images, labels
#    return np.array(images), np.array(labels)

  def extract_dataset_prop(self, n_images,n_images_validation):
    """Loads images of input and its respective labels and returns them into two 4D uint8 numpy array [index, y, x, depth]."""
    print 'Loading images...'
    images = []
    labels = []
    val_images = []
    val_labels = []

    im_loaded = 0
    

    #name = self.im_names[self.tracker]

    # Extract first the training images

    while im_loaded < n_images:


      """End of an epoch"""
      if self.tracker >= len(self.im_names):
        self.tracker = 0
        random.shuffle(self.im_names)

      name = self.im_names[self.tracker]
      #print 'Opening image: ', name
      im = Image.open(name)

      """Replacing names to open its ground truth"""
      repls = ('Training', 'GroundTruth'), ('', '')
      name = reduce(lambda a, kv: a.replace(*kv), repls, name)
        
      lb = Image.open(name) 
      #im = HSVColor(im)
      #lb = HSVColor(lb)

      im = im.resize((512, 512), Image.NEAREST)
      lb = lb.resize((512, 512), Image.NEAREST)

      im = np.asarray(im,dtype=np.float32)
      lb = np.asarray(lb,dtype=np.float32)
      #assert im.shape == lb.shape

      #print im.dtype

      # CUT in patches based on parameters


      images.append(im)
      labels.append(lb)

      im_loaded += 1
      self.tracker += 1

      #print im_loaded
      if im_loaded%500 == 0:
        print 'Loaded ' + str(im_loaded) + ' images.'


    im_loaded = 0
    """Extract the validation images"""


    while im_loaded < n_images_validation:


      """End of an epoch"""
      if self.tracker_val >= len(self.im_names_val):
        self.tracker_val = 0
        random.shuffle(self.im_names_val)

      name = self.im_names_val[self.tracker_val]
      #print 'Opening image: ', name
      im = Image.open(name)

      """Replacing names to open its ground truth"""
      repls = ('Validation', 'ValidationGroundTruth'), ('', '')
      name = reduce(lambda a, kv: a.replace(*kv), repls, name)
        
      lb = Image.open(name) 
      #im = HSVColor(im)
      #lb = HSVColor(lb)
      im = im.resize((512, 512), Image.NEAREST)
      lb = lb.resize((512, 512), Image.NEAREST)

      im = np.asarray(im)
      lb = np.asarray(lb)
      #assert im.shape == lb.shape

      # CUT in patches based on parameters

      

      val_images.append(im)
      val_labels.append(lb)

      im_loaded += 1
      self.tracker_val += 1

      #print im_loaded
      if im_loaded%500 == 0:
        print 'Loaded ' + str(im_loaded) + ' validation images.'


#################################################################
#acho que e assim e mais rapido, tem que testar se funciona mesmo
#################################################################
    return images, labels, val_images, val_labels

  def save_data_sets(self):
    
    if not os.path.exists(array_path):
     os.mkdir(array_path)
    for i in range(0, len(self.im_names)):
            #print ("%d" %len(self.im_names))
	    train_images, train_labels, _, _ = self.extract_dataset_prop(1,0)

	    """shuffling inputs"""
	    if n_images >0:
	      shuffler = list(zip(train_images, train_labels))
	      random.shuffle(shuffler)
	      train_images, train_labels = zip(*shuffler)

	    # test_images = np.array(train_images[:TEST_SIZE])
	    # test_labels = np.array(train_labels[:TEST_SIZE])

	    train_images = np.array(train_images[:])
	    train_labels = np.array(train_labels[:])


	    images_file = open(array_path+"images_"+str(i+1)+".npy", "wb")
	    labels_file = open(array_path+"labels_"+str(i+1)+".npy", "wb")

	    np.save(images_file, train_images)
	    np.save(labels_file, train_labels)
	    
	    images_file.close()
	    labels_file.close()

            print ("%d imagens convertidas" %i)

    for i in range(0, len(self.im_names_val)):
	    _, _, valid_images, valid_labels = self.extract_dataset_prop(0,1)

            valid_images = np.array(valid_images[:])
	    valid_labels = np.array(valid_labels[:])

	    val_images_file = open(array_path+"val_images_"+str(i+1)+".npy", "wb")
	    val_labels_file = open(array_path+"val_labels_"+str(i+1)+".npy", "wb")

            np.save(val_images_file, valid_images)
	    np.save(val_labels_file, valid_labels)

	    val_images_file.close()
	    val_labels_file.close()

            print ("%d imagens de validation convertidas" %i)

converter = DataSetConverter(path,val_path, input_size, proportions, n_images_dataset)
converter.save_data_sets()
    
