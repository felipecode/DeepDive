import tensorflow as tf
import numpy as np
from random import randint
from config import *
import random
import string 


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):

	return ''.join(random.choice(chars) for _ in range(size))






def weight_variable_scaling(shape):
	initializer = tf.uniform_unit_scaling_initializer(factor=0.8)
	initial = tf.get_variable(name=id_generator(), shape=shape, initializer=initializer, trainable=True)
	return initial


def weight_xavi_init(shape):

	initial = tf.get_variable(name=id_generator(), shape=shape,initializer=tf.contrib.layers.xavier_initializer())
	return initial


def bias_variable( shape):  
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)



def extract_features(tf, x, input_size):


	""" Convolution number one  """

	print x
	x = tf.reshape(x, [-1, 32, 32, 3])
	print x


	

	k_h = 11; k_w = 11; c_o = 96; s_h = 1; s_w = 1  


	W_conv1 = weight_xavi_init([k_h,k_w,3,c_o])

	W_b_conv1 = bias_variable([c_o])

	#tf.Variable(net_data["conv1"][1],name='Alex_W_b_conv1')

	#x_drop = tf.nn.dropout(x, dropout[0])

	conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, s_h, s_w, 1], padding="SAME")+ W_b_conv1 )
	
	mu,sigma = tf.nn.moments(conv1,[0,1,2])
	conv1 = tf.nn.batch_normalization(conv1,mu,sigma,None,None,0.01)
	#conv1_drop = tf.nn.dropout(conv1, dropout[1])

	print conv1
	""" Normalization number 1 """
	#lrn1
	#lrn(2, 2e-05, 0.75, name='norm1')
	#radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	#lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

	#maxpool1
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
	print maxpool1

	""" Convolution number two  """
	#conv(5, 5, 256, 1, 1, group=2, name='conv2')
	k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2


	W_conv2 = weight_xavi_init([k_h,k_w,96,c_o])
	W_b_conv2 = bias_variable([c_o])


	
	conv2 = tf.nn.relu(tf.nn.conv2d(maxpool1, W_conv2, strides=[1, s_h, s_w, 1], padding="SAME")+ W_b_conv2 )
	print conv2

	mu,sigma = tf.nn.moments(conv2,[0,1,2])
	conv2 = tf.nn.batch_normalization(conv2,mu,sigma,None,None,0.01)
	#conv2_drop = tf.nn.dropout(conv2, dropout[2])

	#lrn2
	#lrn(2, 2e-05, 0.75, name='norm2')
	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius,alpha=alpha,beta=beta,bias=bias)

	#maxpool2
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

	print maxpool2

	""" Convolution number three  """
	#conv(3, 3, 384, 1, 1, name='conv3')
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
	W_conv3 = weight_xavi_init([k_h,k_w,256,c_o])
	W_b_conv3 = bias_variable([c_o])
	
	
	conv3 = tf.nn.relu(tf.nn.conv2d(maxpool2, W_conv3, strides=[1, s_h, s_w, 1], padding="SAME")+ W_b_conv3 )
	print conv3
	mu,sigma = tf.nn.moments(conv3,[0,1,2])
	conv3 = tf.nn.batch_normalization(conv3,mu,sigma,None,None,0.01)
	#conv3_drop = tf.nn.dropout(conv3, dropout[3])

	""" Convolution number four  """
	#conv(3, 3, 384, 1, 1, group=2, name='conv4')
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
	W_conv4 = weight_xavi_init([k_h,k_w,384,c_o])
	W_b_conv4 = bias_variable([c_o])
	conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W_conv4, strides=[1, s_h, s_w, 1], padding="SAME")+ W_b_conv4 )
	print conv4
	mu,sigma = tf.nn.moments(conv4,[0,1,2])
	conv4 = tf.nn.batch_normalization(conv4,mu,sigma,None,None,0.01)
	#conv4_drop = tf.nn.dropout(conv4, dropout[4])
	#conv5
	#conv(3, 3, 256, 1, 1, group=2, name='conv5')
	k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
	W_conv5 = weight_xavi_init([k_h,k_w,384,c_o])
	W_b_conv5 = bias_variable([c_o])
	conv5 = tf.nn.relu(tf.nn.conv2d(conv4, W_conv5, strides=[1, s_h, s_w, 1], padding="SAME")+ W_b_conv5 )




	mu,sigma = tf.nn.moments(conv5,[0,1,2])
	conv5 = tf.nn.batch_normalization(conv5,mu,sigma,None,None,0.01)
	print conv5

	return conv1,conv2,conv3,conv4,conv5
	# conv5_drop = tf.nn.dropout(conv5, dropout[5])

	# #maxpool5
	# #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
	# k_h = 3; k_w = 3; s_h = 1; s_w = 1; padding = 'VALID'
	# maxpool5 = tf.nn.max_pool(conv5_drop, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

	# #fc6
	# #fc(4096, name='fc6')

	# print maxpool5

	# fc6W = weight_xavi_init([int(np.prod(maxpool5.get_shape()[1:])),4096],name='Alex_W_fc6')
	# fc6b = bias_variable([4096],name='Alex_W_b_fc6')

	# print np.prod(maxpool5.get_shape()[1:])

	# print maxpool5.get_shape()[0]

	# fc6 = tf.nn.relu(tf.nn.xw_plus_b(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b,name='Alex_fc6'))

	# fc6_drop = tf.nn.dropout(fc6, dropout[6])

	# print fc6
	# #fc7
	# #fc(4096, name='fc7')
	# fc7W = weight_xavi_init([4096,4096],name='Alex_W_fc7')
	# fc7b = bias_variable([4096],name='Alex_W_b_fc7')
	# fc7 = tf.nn.relu(tf.nn.xw_plus_b(fc6_drop, fc7W, fc7b, name='Alex_fc7'))
	# #fc7 = tf.nn.xw_plus_b(fc6, fc7W, fc7b, name='Alex_fc7')
	# #fc8
	# #fc(1000, relu=False, name='fc8')
	# fc7_drop = tf.nn.dropout(fc7, dropout[7])


	# fc8W = weight_xavi_init([4096,18],name='Alex_W_fc8')
	# fc8b = bias_variable([18],name='Alex_W_b_fc8')
	# fc8 = tf.nn.xw_plus_b(fc7_drop, fc8W, fc8b, name='Alex_fc8')

	# print fc8

	#return fc8

