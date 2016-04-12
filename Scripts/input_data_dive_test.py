
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
from time import time



class DataSet(object):
  def __init__(self, images_names, labels_names,input_size,output_size):

    #assert len(images_names) == len(labels_names), (
    #    'images.shape: %s labels.shape: %s' % (len(images_names),
    #                                           len(labels_names))


    self._num_examples = len(images_names)
    self._images_names = images_names
    self._labels_names = labels_names
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._input_size= input_size
    self._output_size=output_size


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

  def read_image(self,image_name):
    image =Image.open(image_name)
    image =  image.resize((self._input_size[0], self._input_size[1]), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    return image

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
      perm = np.arange(len(self._images_names))
      np.random.shuffle(perm)
      copy_images_names=list(self._images_names)
      copy_labels_names=list(self._labels_names)
      for n in range(0,len(self._images_names)):
        copy_images_names[n] = self._images_names[perm[n]]
        copy_labels_names[n] = self._labels_names[perm[n]]
      for n in range(0,len(self._images_names)):
        self._images_names[n] = copy_images_names[perm[n]]
        self._labels_names[n] = copy_labels_names[perm[n]]

      


      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    
    
    if batch_size >  (self._num_examples - self._index_in_epoch):

      batch_size = self._num_examples - self._index_in_epoch

      
    images = np.empty((batch_size, self._input_size[0], self._input_size[1],self._input_size[2]))
    if len(self._output_size) > 2:
      labels = np.empty((batch_size, self._output_size[0], self._output_size[1],self._output_size[2]))
    else:
      labels = np.empty((batch_size, self._output_size[0], self._output_size[1]))

    for n in range(batch_size):
      #t0 = time()
      images[n,:,:] = self.read_image(self._images_names[start+n])
      #print time() - t0
      labels[n,:,:] = self.read_image(self._labels_names[start+n])


    return images, labels

 




class DataSetManager(object):

  def __init__(self, path, path_val, path_truth, path_val_truth, input_size,output_size, proportions):
    self.input_size = input_size
    self.output_size =output_size

    """ Get all the image names for training images on a path folder """
    self.im_names  = glob.glob(path + "/*.jpg")
    self.im_names_labels = glob.glob(path_truth + "/*.jpg")

    """ Shufling all the Images with a single permutation """
    perm = np.arange(len(self.im_names))
    np.random.shuffle(perm)

    for n in range(0,len(self.im_names)):
      self.im_names[n] = self.im_names[perm[n]]
      self.im_names_labels[n] = self.im_names_labels[perm[n]]    


    self.im_names_val = glob.glob(path_val + "/*.jpg")
    self.im_names_val_labels = glob.glob(path_val_truth + "/*.jpg")

      

    #random.shuffle(self.im_names_val)

    self.train = DataSet(self.im_names, self.im_names_labels,input_size,output_size)
    self.validation = DataSet(self.im_names_val, self.im_names_val_labels,input_size,output_size)

 
