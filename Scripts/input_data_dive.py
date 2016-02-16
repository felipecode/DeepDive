
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
import Image
from scipy import misc
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import glob

import matplotlib.pyplot as plt

class DataSet(object):
  def __init__(self, images, labels):

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    #assert images.shape[3] == 1
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

  def __init__(self, path, input_size, proportions, n_images_dataset):
    self.input_size = input_size
    self.im_names = []
    self.tracker = 0

    """ First thing we do is to elect a folder number based on the proportions vec """
    #proportions
    for i in range(0,len(proportions)):
      n_images_fold = int(proportions[i]*n_images_dataset)

      path_folder = path + '/' + str(i+1)

      """Replace possible // for /"""
      repls = ('//', '/'), ('', '')
      path_folder = reduce(lambda a, kv: a.replace(*kv), repls, path_folder)

      im_names_fold = glob.glob(path_folder + "/*.jpg")

      #random.shuffle(im_names_fold)
      im_names_fold = im_names_fold[0:n_images_fold]

      self.im_names = self.im_names + im_names_fold
      
    #random.shuffle(self.im_names)



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

    return np.array(images), np.array(labels)

  def extract_dataset_prop(self, n_images):
    """Loads images of input and its respective labels and returns them into two 4D uint8 numpy array [index, y, x, depth]."""
    print 'Loading images...'
    images = []
    labels = []

    im_loaded = 0
    

    #name = self.im_names[self.tracker]
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

      im = np.asarray(im)
      lb = np.asarray(lb)
      #assert im.shape == lb.shape

      images.append(im)
      labels.append(lb)

      im_loaded += 1
      self.tracker += 1

      #print im_loaded
      if im_loaded%500 == 0:
        print 'Loaded ' + str(im_loaded) + ' images.'


    return np.array(images), np.array(labels)


  def read_data_sets(self, n_images):
    
    class DataSets(object):
      pass

    data_sets = DataSets()

    TEST_SIZE = 0
    VALIDATION_SIZE = 100

    train_images, train_labels = self.extract_dataset_prop(n_images)

    """shuffling inputs"""
    shuffler = list(zip(train_images, train_labels))
    random.shuffle(shuffler)
    train_images, train_labels = zip(*shuffler)

    # test_images = np.array(train_images[:TEST_SIZE])
    # test_labels = np.array(train_labels[:TEST_SIZE])

    valid_images = np.array(train_images[TEST_SIZE:VALIDATION_SIZE])
    valid_labels = np.array(train_labels[TEST_SIZE:VALIDATION_SIZE])

    train_images = np.array(train_images[TEST_SIZE+VALIDATION_SIZE:])
    train_labels = np.array(train_labels[TEST_SIZE+VALIDATION_SIZE:])

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(valid_images, valid_labels)
    # data_sets.test  = DataSet(test_images, test_labels)

    return data_sets

    