"""
Returns the last tensor of the network's structure
followed dropout and features dictionaries to be summarised
Input is tensorflow class and an input placeholder.  
"""
import numpy as np
from deep_dive import DeepDive
def create_discriminator_structure(tf, x, input_size): 
  """Deep dive libs"""    
  deep_dive = DeepDive()

  x_image=x
  x_image = tf.contrib.layers.batch_norm(x_image,center=True,updates_collections=None,scale=True,is_training=True)
  W_conv1 = deep_dive.weight_variable_scaling([3,3,3,1],name='W_conv1')
  conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=True)
  #conv1 = deep_dive.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME')
  conv1_pooled=tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name="conv1_pooled")
  shape = int(np.prod(conv1_pooled.get_shape()[1:]))
  conv1_pooled_flat = tf.reshape(conv1_pooled, [-1, shape])
  W_fc1 = tf.get_variable("fc_weights",shape=[shape, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-1))
  b_fc1 = tf.get_variable("fc_biases", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
  fc1 = tf.matmul(conv1_pooled_flat, W_fc1) + b_fc1
  return tf.nn.sigmoid(fc1)

