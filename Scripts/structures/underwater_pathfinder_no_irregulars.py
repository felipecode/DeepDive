def create_structure(tf, x, input_size, dropout):
  from deep_dive import DeepDive

  deep_dive = DeepDive()

  x_image = x
  print x
  dropoutDict={}
  features={}
  scalars={}
  histograms={}

  #first convolution
  #INPUT: 16x16x3
  #KERNEL: 3x3
  #PADDING: 0
  #TIMES APPLIED: 12
  #OUTPUT: 14x14x12
  W_conv1 = deep_dive.weight_variable_scaling([3, 3, 3, 12], name='W_conv1')
  b_conv1 = deep_dive.bias_variable([12])

  histograms["W_conv1"] = W_conv1
  histograms["b_conv1"] = b_conv1

  conv1 = deep_dive.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv1
  features["conv1"] = conv1

  #second convolution
  #INPUT: 14x14x12
  #KERNEL: 3x3
  #PADDING: 0
  #TIMES APPLIED: 16
  #OUTPUT: 12x12x16

  W_conv2 = deep_dive.weight_variable_scaling([3, 3, 4, 16], name='W_conv2')
  b_conv2 = deep_dive.bias_variable([16])

  histograms["W_conv2"] = W_conv2
  histograms["b_conv2"] = b_conv2

  conv2 = deep_dive.conv2d(conv1, W_conv2, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv2
  features["conv2"] = conv2

  #third convolution
  #INPUT: 12x12x16
  #KERNEL: 5x5
  #PADDING: 0
  #TIMES APPLIED: 32
  #OUTPUT: 8x8x32

  W_conv3 = deep_dive.weight_variable_scaling([5, 5, 16, 32], name='W_conv3')
  b_conv3 = deep_dive.bias_variable([32])

  histograms["W_conv3"] = W_conv3
  histograms["b_conv3"] = b_conv3

  conv3 = deep_dive.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv3
  features["conv3"] = conv3

  #first maxpool
  #INPUT: 8x8x32
  #KERNEL: 2x2
  #PADDING: 0
  #OUTPUT: 4x4x32 

  pool1 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')
  print pool1
  features["pool1"] = pool1

  #fourth convolution
  #INPUT: 4x4x32
  #KERNEL: 3x3
  #PADDING: 0
  #TIMES APPLIED: 1
  #OUTPUT: 2x2x1

  W_conv4 = deep_dive.weight_variable_scaling([3, 3, 32, 1], name='W_conv4')
  b_conv4 = deep_dive.bias_variable([1])

  histograms["W_conv4"] = W_conv4
  histograms["b_conv4"] = b_conv4

  conv4 = deep_dive.conv2d(pool1, W_conv4, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv4
  features["conv4"] = conv4

  #first avgpool
  #INPUT: 2x2x1
  #KERNEL: 2x2
  #PADDING: 0
  #OUTPUT: 1x1x1

  pool2 = tf.nn.avg_pool(conv4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')
  print pool2
  features["pool2"] = pool2

  one_constant = tf.constant(1)

  brelu = tf.minimum(tf.to_float(one_constant), tf.nn.relu(pool2, name = "relu"), name = "brelu")
  print brelu

  return brelu, dropoutDict, features, scalars, histograms