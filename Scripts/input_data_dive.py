
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
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
def extract_images(path, max_im, max_y, max_x):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print 'Loading images...'
  images = []
  for i in range(2, max_im):
    for j in range(1, max_y):
      for k in range(1, max_x):
        #open train
        im = Image.open(path + 'Training/i' + str(i) + 'x' + str(k) + 'y' + str(j) + '.png').convert('RGB')
        im = np.array(im)
        images.append(im)


  return np.array(images)

def extract_single_image(path):
  """Extract the image and return a 3D uint8 numpy array [y, x, depth]."""
  print 'Loading image...'
  im = Image.open(path).convert('RGB')

  return np.array(im)

def extract_labels(path, max_im, max_y, max_x, label_size):
  """Extract the labels into a 4D uint8 numpy array [index, y, x, depth]."""
  labels = []
  print 'Loading labels...'
  for i in range(2, max_im):
    for j in range(1, max_y):
      for k in range(1, max_x):
        #open gt
        label = Image.open(path + 'GroundTruth/i1' + 'x' + str(k) + 'y' + str(j) + '.png').convert('RGB')

        label = label.resize(label_size, Image.ANTIALIAS)
        label = np.array(label)
        labels.append(label)

  return np.array(labels)

class DataSet(object):
  def __init__(self, images, labels, fake_data=False, one_hot=False):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

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
  def next_batch(self, batch_size, fake_data=False):
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
def read_data_sets(label_size):
  class DataSets(object):
    pass
  data_sets = DataSets()


  VALIDATION_SIZE = 500

  path = '/home/nautec/Downloads/TURBID/Photo3D/'

  train_images = extract_images(path , max_im=5, max_y=58, max_x=40)
  train_labels = extract_labels(path , max_im=5, max_y=58, max_x=40, label_size=label_size)

  """shuffling inputs"""
  shuffler = list(zip(train_images, train_labels))
  random.shuffle(shuffler)
  train_images, train_labels = zip(*shuffler)

  train_images = np.array(train_images)
  train_labels = np.array(train_labels)

  #train_images = np.array([extract_single_image(path + 'imagepatch.png')])

  #train_labels = np.array([extract_single_image(path + 'imagepatch.png')])



  # test_images = extract_images(path)
  # test_labels = extract_labels(path)

  #validation_images = train_images[:VALIDATION_SIZE]
  #validation_labels = train_labels[:VALIDATION_SIZE]

  #train_images = train_images[VALIDATION_SIZE:]
  #train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels)
  #data_sets.validation = DataSet(validation_images, validation_labels)
  # data_sets.test = DataSet(test_images, test_labels)

  return data_sets