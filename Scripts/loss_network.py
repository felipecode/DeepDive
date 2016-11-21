"""
	loss_network.py

	This file implements a VGG16 Loss Network which calculates the feature reconstruction loss based on the difference
	of feature maps from a pre-selected convolution.
"""
import numpy as np
from config import *
config=configMain()
# A adicionar em config.py
WEIGHTS_FILE = config.WEIGHTS_FILE

def load_loss_weights(loss_parameters, sess):
	weights = np.load(WEIGHTS_FILE)
	keys = sorted(weights.keys())
	for i, k in enumerate(keys):
		sess.run(loss_parameters[i].assign(weights[k]))

def create_loss_structure(tf, x_image, y_image, sess):
	loss			= 0.0
	loss_parameters 	= []
	
	# Preprocess
	mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name="img_mean")
	x_image -= mean
	y_image -= mean

	# First Convolution
	W_conv11 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv11 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv11, b_conv11]

	# Output
	conv11 = tf.nn.conv2d(x_image, W_conv11, strides=[1, 1, 1, 1], padding='SAME') + b_conv11
	conv11 = tf.nn.relu(conv11)

	# Ground-Truth
	conv11_gt = tf.nn.conv2d(y_image, W_conv11, strides=[1, 1, 1, 1], padding='SAME') + b_conv11
	conv11_gt = tf.nn.relu(conv11_gt)
	
	# Loss
	# ...
	
	# Second Convolution
	W_conv12 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv12 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv12, b_conv12]

	# Output
	conv12 = tf.nn.conv2d(conv11, W_conv12, strides=[1, 1, 1, 1], padding='SAME') + b_conv12
	conv12 = tf.nn.relu(conv12)

	# Ground-Truth
	conv12_gt = tf.nn.conv2d(conv11_gt, W_conv12, strides=[1, 1, 1, 1], padding='SAME') + b_conv12
	conv12_gt = tf.nn.relu(conv12_gt)

	# Loss
	# ...
	
	# First Maxpool
	pool1 = tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")
	pool1_gt = tf.nn.max_pool(conv12_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1_gt")
	
	# Third Convolution
	W_conv21 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv21 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv21, b_conv21]

	# Output
	conv21 = tf.nn.conv2d(pool1, W_conv21, strides=[1, 1, 1, 1], padding='SAME') + b_conv21
	conv21 = tf.nn.relu(conv21)

	# Ground-Truth
	conv21_gt = tf.nn.conv2d(pool1_gt, W_conv21, strides=[1, 1, 1, 1], padding='SAME') + b_conv21
	conv21_gt = tf.nn.relu(conv21_gt)

	# Loss
	# ...
	
	# Fourth Convolution
	W_conv22 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv22 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv22, b_conv22]

	# Output
	conv22 = tf.nn.conv2d(conv21, W_conv22, strides=[1, 1, 1, 1], padding='SAME') + b_conv22
	conv22 = tf.nn.relu(conv22)

	# Ground-Truth
	conv22_gt = tf.nn.conv2d(conv21_gt, W_conv22, strides=[1, 1, 1, 1], padding='SAME') + b_conv22
	conv22_gt = tf.nn.relu(conv22_gt)

	# Loss
	norm = tf.to_float(conv22.get_shape()[1] * conv22.get_shape()[2] * conv22.get_shape()[3])
	loss += tf.div(tf.reduce_sum(tf.squared_difference(conv22, conv22_gt), reduction_indices=[1,2,3]), norm)
	
	# Second Maxpool
	pool2 = tf.nn.max_pool(conv22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")
	pool2_gt = tf.nn.max_pool(conv22_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2_gt")
	
	# Fifth Convolution
	W_conv31 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv31 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv31, b_conv31]

	# Output
	conv31 = tf.nn.conv2d(pool2, W_conv31, strides=[1, 1, 1, 1], padding='SAME') + b_conv31
	conv31 = tf.nn.relu(conv31)

	# Ground-Truth
	conv31_gt = tf.nn.conv2d(pool2_gt, W_conv31, strides=[1, 1, 1, 1], padding='SAME') + b_conv31
	conv31_gt = tf.nn.relu(conv31_gt)

	# Loss
	# ...
	
	# Sixth Convolution
	W_conv32 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv32 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv32, b_conv32]

	# Output
	conv32 = tf.nn.conv2d(conv31, W_conv32, strides=[1, 1, 1, 1], padding='SAME') + b_conv32
	conv32 = tf.nn.relu(conv32)

	# Ground-Truth
	conv32_gt = tf.nn.conv2d(conv31_gt, W_conv32, strides=[1, 1, 1, 1], padding='SAME') + b_conv32
	conv32_gt = tf.nn.relu(conv32_gt)

	# Loss
	# ...
	
	# Seventh Convolution
	W_conv33 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv33 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv33, b_conv33]

	# Output
	conv33 = tf.nn.conv2d(conv32, W_conv33, strides=[1, 1, 1, 1], padding='SAME') + b_conv33
	conv33 = tf.nn.relu(conv33)

	# Ground-Truth
	conv33_gt = tf.nn.conv2d(conv32_gt, W_conv33, strides=[1, 1, 1, 1], padding='SAME') + b_conv33
	conv33_gt = tf.nn.relu(conv33_gt)

	# Loss
	# ...
	
	# Third Maxpool
	pool3 = tf.nn.max_pool(conv33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")
	pool3_gt = tf.nn.max_pool(conv33_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3_gt")
	
	# Eighth Convolution
	W_conv41 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv41 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv41, b_conv41]

	# Output
	conv41 = tf.nn.conv2d(pool3, W_conv41, strides=[1, 1, 1, 1], padding='SAME') + b_conv41
	conv41 = tf.nn.relu(conv41)

	# Ground-Truth
	conv41_gt = tf.nn.conv2d(pool3_gt, W_conv41, strides=[1, 1, 1, 1], padding='SAME') + b_conv41
	conv41_gt = tf.nn.relu(conv41_gt)
	
	# Loss
	# ...
	
	# Nineth Convolution
	W_conv42 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv42 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv42, b_conv42]

	# Output
	conv42 = tf.nn.conv2d(conv41, W_conv42, strides=[1, 1, 1, 1], padding='SAME') + b_conv42
	conv42 = tf.nn.relu(conv42)

	# Ground-Truth
	conv42_gt = tf.nn.conv2d(conv41_gt, W_conv42, strides=[1, 1, 1, 1], padding='SAME') + b_conv42
	conv42_gt = tf.nn.relu(conv42_gt)
	
	# Loss
	# ...
	
	# Tenth Convolution
	W_conv43 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv43 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv43, b_conv43]

	# Output
	conv43 = tf.nn.conv2d(conv42, W_conv43, strides=[1, 1, 1, 1], padding='SAME') + b_conv43
	conv43 = tf.nn.relu(conv43)

	# Ground-Truth
	conv43_gt = tf.nn.conv2d(conv42_gt, W_conv43, strides=[1, 1, 1, 1], padding='SAME') + b_conv43
	conv43_gt = tf.nn.relu(conv43_gt)
	
	# Loss
	# ...
	
	# Fourth Maxpool
	pool4 = tf.nn.max_pool(conv43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")
	pool4_gt = tf.nn.max_pool(conv43_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4_gt")
	
	# Eleventh Convolution
	W_conv51 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv51 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv51, b_conv51]

	# Output
	conv51 = tf.nn.conv2d(pool4, W_conv51, strides=[1, 1, 1, 1], padding='SAME') + b_conv51
	conv51 = tf.nn.relu(conv51)

	# Ground-Truth
	conv51_gt = tf.nn.conv2d(pool4_gt, W_conv51, strides=[1, 1, 1, 1], padding='SAME') + b_conv51
	conv51_gt = tf.nn.relu(conv51_gt)

	# Loss
	# ...
	
	# Twelfth Convolution
	W_conv52 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv52 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv52, b_conv52]

	# Output
	conv52 = tf.nn.conv2d(conv51, W_conv52, strides=[1, 1, 1, 1], padding='SAME') + b_conv52
	conv52 = tf.nn.relu(conv52)
	
	# Ground-Truth
	conv52_gt = tf.nn.conv2d(conv51_gt, W_conv52, strides=[1, 1, 1, 1], padding='SAME') + b_conv52
	conv52_gt = tf.nn.relu(conv52_gt)

	# Loss
	# ...

	# Thirteenth Convolution
	W_conv53 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_conv53 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_conv53, b_conv53]

	# Output
	conv53 = tf.nn.conv2d(conv52, W_conv53, strides=[1, 1, 1, 1], padding='SAME') + b_conv53
	conv53 = tf.nn.relu(conv53)

	# Ground-Truth
	conv53_gt = tf.nn.conv2d(conv52_gt, W_conv53, strides=[1, 1, 1, 1], padding='SAME') + b_conv53
	conv53_gt = tf.nn.relu(conv53_gt)

	# Loss
	# ...
	
	# Fifth Maxpool
	pool5 = tf.nn.max_pool(conv53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5")
	pool5_gt = tf.nn.max_pool(conv53_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5_gt")
	
	# FC Parameters
	shape = int(np.prod(pool5.get_shape()[1:]))
	pool5_flat = tf.reshape(pool5, [-1, shape])
	pool5_gt_flat = tf.reshape(pool5_gt, [-1, shape])
	
	# First FC
	W_fc1 = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_fc1 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_fc1, b_fc1]

	# Output
	fc1 = tf.matmul(pool5_flat, W_fc1) + b_fc1
	fc1 = tf.nn.relu(fc1)

	# Ground-Truth
	fc1_gt = tf.matmul(pool5_gt_flat, W_fc1) + b_fc1
	fc1_gt = tf.nn.relu(fc1_gt)

	# Loss
	# ...
	
	# Second FC
	W_fc2 = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_fc2 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_fc2, b_fc2]

	# Output
	fc2 = tf.matmul(fc1, W_fc2) + b_fc2
	fc2 = tf.nn.relu(fc2)
	
	# Ground-Truth
	fc2_gt = tf.matmul(fc1_gt, W_fc2) + b_fc2
	fc2_gt = tf.nn.relu(fc2_gt)

	# Loss
	# ...

	# Third FC
	W_fc3 = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32, stddev=1e-1), trainable=False, name="weights")
	b_fc3 = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32), trainable=False, name="biases")
	loss_parameters += [W_fc3, b_fc3]

	# Output
	fc3 = tf.matmul(fc2, W_fc3) + b_fc3

	# Ground-Truth
	fc3_gt = tf.matmul(fc2_gt, W_fc3) + b_fc3

	# Loss
	# ...
	
	# First Softmax
	softmax = tf.nn.softmax(fc3)
	softmax_gt = tf.nn.softmax(fc3_gt)
	
	# Load Weights
	load_loss_weights(loss_parameters, sess)
	
	return loss