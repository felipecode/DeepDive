
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


def HSVColor(img):
  if isinstance(img,Image.Image):
      r,g,b = img.split()
      Hdat = []
      Sdat = []
      Vdat = [] 
      for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
          h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
          Hdat.append(int(h*255.))
          Sdat.append(int(s*255.))
          Vdat.append(int(v*255.))
      r.putdata(Hdat)
      g.putdata(Sdat)
      b.putdata(Vdat)
      return Image.merge('RGB',(r,g,b))
  else:
      return None






class DataSet(object):
  def __init__(self, images, labels):

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    #assert images.shape[3] == 1
    #print 'shapes'
    #print images.shape
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2] * images.shape[3])
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)

    labels = labels.reshape(labels.shape[0],
                            labels.shape[1] * labels.shape[2] * labels.shape[3])
    # Convert from [0, 255] -> [0.0, 1.0].
    labels = labels.astype(np.float32)
    labels = np.multiply(labels, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

class DataSetManager(object):

  def __init__(self, path, path_val, input_size, proportions, n_images_dataset):
    self.input_size = input_size
    self.im_names = []
    self.im_names_val = []
    self.tracker = 0
    self.tracker_val = 0
    n_images_dataset_val = int(n_images_dataset*0.1)

    """ First thing we do is to elect a folder number based on the proportions vec """
    #proportions
    for i in range(0,len(proportions)):
      n_images_fold = int(proportions[i]*n_images_dataset)
      n_images_fold_val = int(proportions[i]*n_images_dataset_val)

      path_folder = path + '/' + str(i+1)
      path_folder_val = path_val + '/' + str(i+1)

      """Replace possible // for /"""
      repls = ('//', '/'), ('', '')
      path_folder = reduce(lambda a, kv: a.replace(*kv), repls, path_folder)
      path_folder_val = reduce(lambda a, kv: a.replace(*kv), repls, path_folder)

      im_names_fold = glob.glob(path_folder + "/*.jpg")
      im_names_fold_val = glob.glob(path_folder_val + "/*.jpg")

      #random.shuffle(im_names_fold)
      im_names_fold = im_names_fold[0:n_images_fold]
      im_names_fold_val = im_names_fold_val[0:n_images_fold_val]

      self.im_names = self.im_names + im_names_fold
      self.im_names_val = self.im_names_val + im_names_fold_val
      
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

      repls = ('Training', 'GroundTruth'), ('', '')
      name = reduce(lambda a, kv: a.replace(*kv), repls, name)
        
      lb = Image.open(name) 

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

      im = np.asarray(im)
      lb = np.asarray(lb)
      #assert im.shape == lb.shape

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
#   return np.array(images), np.array(labels), np.array(val_images), np.array(val_labels)

  def read_data_sets(self, n_images,n_images_validation):
    
    class DataSets(object):
      pass

    data_sets = DataSets()

    TEST_SIZE = 0
    VALIDATION_SIZE = 100

    train_images, train_labels, valid_images, valid_labels = self.extract_dataset_prop(n_images,n_images_validation)

    """shuffling inputs"""
    if n_images >0:
      shuffler = list(zip(train_images, train_labels))
      random.shuffle(shuffler)
      train_images, train_labels = zip(*shuffler)

    # test_images = np.array(train_images[:TEST_SIZE])
    # test_labels = np.array(train_labels[:TEST_SIZE])

    valid_images = np.array(valid_images[:])
    valid_labels = np.array(valid_labels[:])

    train_images = np.array(train_images[:])
    train_labels = np.array(train_labels[:])


    """ TODO: take care when the dataset is not valid (no images) """
    if n_images >0:
      data_sets.train = DataSet(train_images, train_labels)
    if n_images_validation >0:
      data_sets.validation = DataSet(valid_images, valid_labels)
    # data_sets.test  = DataSet(test_images, test_labels)

    return data_sets

    
