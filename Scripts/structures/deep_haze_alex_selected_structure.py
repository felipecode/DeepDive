import sys
import os
import cv2
import time
import StringIO
from threading import Lock

sys.path.insert(0, os.path.join('/home/nautec/deep-visualization-toolbox'))
from misc import WithTimer
from core import CodependentThread
from image_misc import norm01, norm01c, norm0255, tile_images_normalize, ensure_float01, tile_images_make_tiles, ensure_uint255_and_resize_to_fit, caffe_load_image, get_tiles_height_width
from image_misc import FormattedString, cv2_typeset_text, to_255

sys.path.insert(0, os.path.join('/home/nautec/caffe', 'python'))
import caffe

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# from tensorflow.python.framework import types
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_nn_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *


def create_structure(tf, x):
	"""Core libs"""
	import numpy as np
	from deep_dive import DeepDive


	deep_dive = DeepDive()
	path = '../Local_aux/weights/'
	
	net = caffe.Classifier(
	            '/home/nautec/deep-visualization-toolbox/models/caffenet-yos/caffenet-yos-deploy.prototxt',
	            '/home/nautec/deep-visualization-toolbox/models/caffenet-yos/caffenet-yos-weights', 
	            
	            #image_dims = (227,227),
	        )

	# weights = []
	# for i in range(0, 96):
	# 	weight = net.params['conv1'][0].data[i]
	# 	weights.append(np.dstack((weight[2],weight[1],weight[0])))
	weight = []
	weight.append(net.params['conv1'][0].data[60])
	weight.append(net.params['conv1'][0].data[61])
	weight.append(net.params['conv1'][0].data[78])
	weight.append(net.params['conv1'][0].data[88])
	weight.append(net.params['conv1'][0].data[74])
	weight.append(net.params['conv1'][0].data[76])
	weight.append(net.params['conv1'][0].data[43])

	weight = np.array(weight)
	weight = np.reshape(weight, [11,11,3,7])

	W_conv1 = tf.Variable(weight, trainable=False)

	b_conv1 = deep_dive.bias_variable([7])

	W_fc1 = deep_dive.weight_variable([87 * 87 * 7, 1024])
	b_fc1 = deep_dive.bias_variable([1024])

	W_fc2 = deep_dive.weight_variable([1024, 2])
	b_fc2 = deep_dive.bias_variable([2])


	"""Structure"""

	x_image = tf.reshape(x, [-1, 184, 184, 3])

	h_conv1 = tf.nn.relu(deep_dive.conv2d(x_image, W_conv1) + b_conv1, name="first_sigmoid")

	h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

	h_pool1_flat = tf.reshape(h_pool1, [-1, 87*87*7])

	h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

	# keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, 0.9)

	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	return y_conv


