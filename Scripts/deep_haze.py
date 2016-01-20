def create_structure(tf, x):
	"""Core libs"""
	import numpy as np
	from deep_dive import DeepDive


	deep_dive = DeepDive()
	path = '../Local_aux/weights/'

	W_conv1 = deep_dive.weight_variable([5,5,3,10])

	W_conv2 = deep_dive.weight_variable([3,3,10,20])

	b_conv1 = deep_dive.bias_variable([10])

	b_conv2 = deep_dive.bias_variable([20])

	W_fc1 = deep_dive.weight_variable([178 * 178 * 20, 20])

	b_fc1 = deep_dive.bias_variable([20])


	W_fc2 = deep_dive.weight_variable([20, 2])
	b_fc2 = deep_dive.bias_variable([2])


	"""Structure"""

	x_image = tf.reshape(x, [-1, 184, 184, 3])

	h_conv1 = tf.nn.relu(deep_dive.conv2d(x_image, W_conv1) + b_conv1, name="first_sigmoid")

	h_conv2 = tf.nn.relu(deep_dive.conv2d(h_conv1, W_conv2) + b_conv2)
	h_conv2_flat = tf.reshape(h_conv2, [-1, 178*178*20])

	h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

	# keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	return y_conv
	# print("test accuracy %g"%accuracy.eval(feed_dict={
	#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

