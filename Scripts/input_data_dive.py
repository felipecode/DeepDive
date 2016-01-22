
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

def extract_dataset(path, input_size):
  """Creates chunkes of inputs and its respective labels and returns them into two 4D uint8 numpy array [index, y, x, depth]."""
  print 'Loading images...'
  images = []
  labels = []
  # for i in range(start_im, max_im):
  #   for j in range(1, max_y):
  #     for k in range(1, max_x):
  #       #open train
  #       im = Image.open(path + 'Training/i' + str(i) + 'x' + str(k) + 'y' + str(j) + '.png').convert('RGB')
  #       im = np.array(im)
  #       images.append(im)
  # for i in range(1, 21):
  #   if i < 8:
  #     im = Image.open(path + str(i) + '.jpg')
  #   elif i < 10:
  #     im = Image.open(path + str(i) + 'pd.jpg')
  #   elif i < 20:
  #     im = Image.open(path + 'a' + str(i) + 'pd.jpg')
  #   else:
  #     im = Image.open(path + 'b' + str(i) + 'pd.jpg')
  im_names =  glob.glob(path + "*.jpg")
  im_opened = 0

  for name in im_names:
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

        im_opened += 1
        if im_opened % 500 == 0:
          print 'Images and labels generated: ', im_opened

  return np.array(images), np.array(labels)




def extract_images(path, max_im, start_im, max_y, max_x):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print 'Loading images...'
  images = []
  # for i in range(start_im, max_im):
  #   for j in range(1, max_y):
  #     for k in range(1, max_x):
  #       #open train
  #       im = Image.open(path + 'Training/i' + str(i) + 'x' + str(k) + 'y' + str(j) + '.png').convert('RGB')
  #       im = np.array(im)
  #       images.append(im)
  # for i in range(1, 21):
  #   if i < 8:
  #     im = Image.open(path + str(i) + '.jpg')
  #   elif i < 10:
  #     im = Image.open(path + str(i) + 'pd.jpg')
  #   elif i < 20:
  #     im = Image.open(path + 'a' + str(i) + 'pd.jpg')
  #   else:
  #     im = Image.open(path + 'b' + str(i) + 'pd.jpg')
  # im_names =   glob.glob(path + "*.jpg")

  #   for name in im_names:
  #     im = Image.open(name)


  #     im = im.resize((1200, 815), Image.ANTIALIAS)

  #   images.append(np.array(im))

  # print images[0]

  return np.array(images)

def extract_single_image(path):
  """Extract the image and return a 3D uint8 numpy array [y, x, depth]."""
  print 'Loading image...'
  im = Image.open(path).convert('RGB')

  return im

def extract_labels(path, start_im, max_im, max_y, max_x, label_size):
  """Extract the labels into a 4D uint8 numpy array [index, y, x, depth]."""
  labels = []
  # print 'Loading labels...'
  # for i in range(start_im, max_im):
  #   for j in range(1, max_y):
  #     for k in range(1, max_x):
  #       #open gt
  #       label = Image.open(path + 'GroundTruth/i1' + 'x' + str(k) + 'y' + str(j) + '.png').convert('RGB')

  #       label = label.resize(label_size, Image.ANTIALIAS)
  #       label = np.array(label)
  #       labels.append(label)
  for i in range(1, 21):
    im = Image.open(path + '1.jpg')
    im = im.resize((1200, 815), Image.ANTIALIAS)
    labels.append(np.array(im))

  return np.array(labels)

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
def read_data_sets(path, input_size):
  
  class DataSets(object):
    pass

  data_sets = DataSets()

  TEST_SIZE = 20
  VALIDATION_SIZE = 100

  train_images, train_labels = extract_dataset(path, input_size)

  """shuffling inputs"""
  shuffler = list(zip(train_images, train_labels))
  random.shuffle(shuffler)
  train_images, train_labels = zip(*shuffler)

  test_images = np.array(train_images[:TEST_SIZE])
  test_labels = np.array(train_labels[:TEST_SIZE])

  valid_images = np.array(train_images[TEST_SIZE:VALIDATION_SIZE])
  valid_labels = np.array(train_labels[TEST_SIZE:VALIDATION_SIZE])

  train_images = np.array(train_images[TEST_SIZE+VALIDATION_SIZE:])
  train_labels = np.array(train_labels[TEST_SIZE+VALIDATION_SIZE:])

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(valid_images, valid_labels)
  data_sets.test  = DataSet(test_images, test_labels)

  return data_sets