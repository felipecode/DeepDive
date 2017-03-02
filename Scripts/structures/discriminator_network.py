import numpy as np

from deep_dive import DeepDive
from discriminator_layer import discriminator_layer

def create_discriminator_structure(tf, x, input_size):
  deep_dive = DeepDive()
  dropoutDict={}
  features={}
  scalars={}
  histograms={}

  x_image = x
  W_conv1 = deep_dive.weight_variable_scaling([3,3,3,64],name='W_conv1')
  b_conv1 = deep_dive.bias_variable([64])
  conv1 = deep_dive.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')

  leaky_relu1 = tf.maximum(0.1*conv1, conv1)

  layer1, features1, histograms1 = discriminator_layer(tf, leaky_relu1, "A", 64, 64, 2, training = True)
  features.update(features1)
  histograms.update(histograms1)

  layer2, features2, histograms2 = discriminator_layer(tf, layer1, "B", 128, 64, 1, training = True)
  features.update(features2)
  histograms.update(histograms2)

  layer3, features3, histograms3 = discriminator_layer(tf, layer2, "C", 128, 128, 2, training = True)
  features.update(features3)
  histograms.update(histograms3)

  layer4, features4, histograms4 = discriminator_layer(tf, layer3, "D", 256, 128, 1, training = True)
  features.update(features4)
  histograms.update(histograms4)

  layer5, features5, histograms5 = discriminator_layer(tf, layer4, "E", 256, 256, 2, training = True)
  features.update(features5)
  histograms.update(histograms5)

  layer6, features6, histograms6 = discriminator_layer(tf, layer5, "F", 512, 256, 2, training = True)
  features.update(features6)
  histograms.update(histograms6)

  layer7, features7, histograms7 = discriminator_layer(tf, layer6, "G", 512, 512, 2, training = True)
  features.update(features7)
  histograms.update(histograms7)

  layer8, features8, histograms8 = discriminator_layer(tf, layer7, "H", 512, 512, 2, training = True)
  features.update(features8)
  histograms.update(histograms8)

  shape = int(np.prod(layer8.get_shape()[1:]))
  layer8_flat = tf.reshape(layer8, [-1, shape])
  W_fc = tf.get_variable("fc_weights",shape=[shape, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-1))
  b_fc = tf.get_variable("fc_biases", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
  fc = tf.matmul(layer8_flat, W_fc) + b_fc

  leaky_relu2 = tf.maximum(0.1*fc, fc)

  return leaky_relu2
