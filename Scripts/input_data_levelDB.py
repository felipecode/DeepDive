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
import leveldb
import matplotlib.pyplot as plt
from time import time
from config import configMain

config=configMain()

def readImageFromDB(db, key, size):
  image =  np.reshape(np.fromstring(db.Get(key),dtype=np.float32),size)
  return image

class DataSet(object):
  def __init__(self, images_key, input_size,output_size, num_examples,db):
    self._db=db
    self._num_examples = num_examples
    self._images_key = images_key
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._input_size= input_size
    self._output_size=output_size

  def getTransmission(n):
    self._db.Get(str(n))

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch >= self._num_examples:
      # Finished epoch
      print 'end epoch'
      self._epochs_completed += 1
      # Shuffle the data
      """ Shufling all the Images with a single permutation """
      random.shuffle(self._images_key)
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    if batch_size >  (self._num_examples - self._index_in_epoch):
      batch_size = self._num_examples - self._index_in_epoch

    images = np.empty((batch_size, self._input_size[0], self._input_size[1],self._input_size[2]))
    labels = np.empty((batch_size, self._output_size[0], self._output_size[1],self._output_size[2]))
    transmission = range(batch_size)
    for n in range(batch_size):
      images[n] = readImageFromDB(self._db,str(self._images_key[start+n]),self._input_size)
      labels[n] = readImageFromDB(self._db,str(self._images_key[start+n])+"label",self._output_size)
      transmission[n] = self._db.Get(str(self._images_key[start+n])+"trans")
    return images, labels, transmission


  def next_batch_normalized(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch >= self._num_examples:
      # Finished epoch
      print 'end epoch'
      self._epochs_completed += 1
      # Shuffle the data
      """ Shufling all the Images with a single permutation """
      random.shuffle(self._images_key)
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    if batch_size >  (self._num_examples - self._index_in_epoch):
      batch_size = self._num_examples - self._index_in_epoch

    images = np.empty((batch_size, self._input_size[0], self._input_size[1],self._input_size[2]))
    labels = np.empty((batch_size, self._output_size[0], self._output_size[1],self._output_size[2]))
    
    for n in range(batch_size):
      images[n] = readImageFromDB(self._db,str(self._images_key[start+n]),self._input_size)
      labels[n] = readImageFromDB(self._db,str(self._images_key[start+n])+"label",self._output_size)
    imagemMedia = np.mean(images, axis=0)

    for n in range(batch_size):
      images[n] = images[n] - imagemMedia
    return images, labels

class DataSetManager(object):

  def __init__(self, path, path_val, path_truth, path_val_truth, input_size,output_size):
    self.input_size = input_size
    self.output_size = output_size
    self.db = leveldb.LevelDB(config.leveldb_path + 'db') 
    self.num_examples = int(self.db.Get('num_examples'))
    self.num_examples_val = int(self.db.Get('num_examples_val'))
    self.images_key = range(self.num_examples)
    self.images_key_val = range(self.num_examples_val)
    for i in range(self.num_examples_val):
      self.images_key_val[i] = 'val' + str(i)
    self.train = DataSet(self.images_key,input_size,output_size,self.num_examples,self.db)
    self.validation = DataSet(self.images_key_val,input_size,output_size,self.num_examples_val,self.db)

  def getNImagesDataset(self):
    return self.num_examples

  def getNImagesValidation(self):
    return self.num_examples_val