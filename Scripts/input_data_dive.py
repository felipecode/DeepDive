
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

def extract_dataset(path, input_size, n_images):
  """Creates chunkes of inputs and its respective labels and returns them into two 4D uint8 numpy array [index, y, x, depth]."""
  print 'Loading images...'
  images = []
  labels = []

  im_names =  glob.glob(path + "*.jpg")
  im_generated = 0
  random.shuffle(im_names)

  for name in im_names:
    if im_generated >= n_images:
      break

    im = Image.open(name)
    repls = ('Ciano_', ''), ('Blue_', ''), ('Green_', ''), ('Training', 'GroundTruth')

    name = reduce(lambda a, kv: a.replace(*kv), repls, name)
    lb = Image.open(name) 

    im = np.asarray(im)
    lb = np.asarray(lb)

    height, width = im.shape[0], im.shape[1]

    for h in range(0, height-input_size[0], input_size[0]):
      for w in range(0, width-input_size[1], input_size[1]):
        h_end = h + input_size[0]
        w_end = w + input_size[1]

        chunk = im[h:h_end, w:w_end]
        images.append(chunk)

        chunk = lb[h: h_end, w:w_end]
        labels.append(chunk)

        im_generated += 1

  return np.array(images), np.array(labels)


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
def read_data_sets(path, input_size, n_images):
  
  class DataSets(object):
    pass

  data_sets = DataSets()

  TEST_SIZE = 0
  VALIDATION_SIZE = 100

  train_images, train_labels = extract_dataset(path, input_size, n_images)

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
  data_sets.test  = DataSet(test_images, test_labels)

  return data_sets