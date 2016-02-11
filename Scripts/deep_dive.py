"""
Implements the DeepDive class.
Use it to create the convolutional neural network architecture.
"""
import tensorflow as tf
import numpy as np
from random import randint
from config import *

class DeepDive(object):

  def __init__(self):
    pass

  """
  Creates a weight variable
  shape: tuple defining the number of weights
  """
  def weight_variable(self, shape, name):
    initial = tf.truncated_normal(shape, stddev=init_std_dev)
    return tf.Variable(initial, name=name)

  def weight_variable_scaling(self, shape, name):
    initializer = tf.uniform_unit_scaling_initializer(factor=1.15)
    initial = tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=True)
    return initial

  """
  Creates a bias variable
  shape: tuple defining the number of biases
  """
  def bias_variable(self, shape, name):  
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

  # def bias_variable(self, shape):  
  #   initial = tf.constant(0.1, shape=shape)
  #   return tf.Variable(initial)

  """
  Creates a 2d Convolution layer.
  x: input layer (tensor)
  padding: 'same' or 'valid'
  W: variable or constant weight created.
  """
  def conv2d(self, x, W, strides=[1,1,1,1], padding='VALID'):
    return tf.nn.conv2d(x, W, strides=strides ,padding=padding)

  """
  Creates a dropout layer.
  x: input layer (tensor)
  keep_prob: probability to keep a node
  """
  def dropout(self, x, keep_prob=0.9):
    seed = randint(0, 100000)
    return tf.nn.dropout(x=x, keep_prob=keep_prob, seed=seed)

  """
  Creates a max pooling layer.
  x: input layer (tensor)
  """
  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
