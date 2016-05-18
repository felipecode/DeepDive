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
  #KERNEL: 3x1
  #PADDING: 0
  #TIMES APPLIED: 16
  #OUTPUT: 14x16x12
  W_conv1 = deep_dive.weight_variable_scaling([3, 1, 3, 16], name='W_conv1')
  b_conv1 = deep_dive.bias_variable([16])

  histograms["W_conv1"] = W_conv1
  histograms["b_conv1"] = b_conv1

  conv1 = deep_dive.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv1
  features["conv1"] = conv1

  #second convolution
  #INPUT: 14x16x12
  #KERNEL: 1x3
  #PADDING: 0
  #TIMES APPLIED: 16
  #OUTPUT: 14x14x16

  W_conv2 = deep_dive.weight_variable_scaling([1, 3, 16, 16], name='W_conv2')
  b_conv2 = deep_dive.bias_variable([16])

  histograms["W_conv2"] = W_conv2
  histograms["b_conv2"] = b_conv2

  conv2 = deep_dive.conv2d(conv1, W_conv2, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv2
  features["conv2"] = conv2

  #third convolution
  #INPUT: 14x14x16
  #KERNEL: 5x1
  #PADDING: 0
  #TIMES APPLIED: 32
  #OUTPUT: 10x14x32

  W_conv3 = deep_dive.weight_variable_scaling([5, 1, 16, 32], name='W_conv3')
  b_conv3 = deep_dive.bias_variable([32])

  histograms["W_conv3"] = W_conv3
  histograms["b_conv3"] = b_conv3

  conv3 = deep_dive.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv3
  features["conv3"] = conv3

  #fourth convolution
  #INPUT: 10x14x32
  #KERNEL: 1x5
  #PADDING: 0
  #TIMES APPLIED: 32
  #OUTPUT: 10x10x32

  W_conv4 = deep_dive.weight_variable_scaling([1, 5, 32, 32], name='W_conv4')
  b_conv4 = deep_dive.bias_variable([32])

  histograms["W_conv4"] = W_conv4
  histograms["b_conv4"] = b_conv4

  conv4 = deep_dive.conv2d(conv3, W_conv4, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv4
  features["conv4"] = conv4

  #first maxpool
  #INPUT: 10x10x32
  #KERNEL: 5x5
  #PADDING: 0
  #OUTPUT: 2x2x32 

  pool1 = tf.nn.max_pool(conv4, ksize = [1, 5, 5, 1], strides = [1, 5, 5, 1], padding = 'VALID', name = 'pool1')
  print pool1
  features["pool1"] = pool1

  #fifth convolution
  #INPUT: 2x2x32
  #KERNEL: 2x2
  #PADDING: 0
  #TIMES APPLIED: 1
  #OUTPUT: 1x1x1

  W_conv5 = deep_dive.weight_variable_scaling([2, 2, 32, 1], name='W_conv5')
  b_conv5 = deep_dive.bias_variable([1])

  histograms["W_conv5"] = W_conv5
  histograms["b_conv5"] = b_conv5

  conv5 = deep_dive.conv2d(pool1, W_conv5, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv5
  features["conv5"] = conv5

  one_constant = tf.constant(1)

  brelu = tf.minimum(tf.to_float(one_constant), tf.nn.relu(conv5, name = "relu"), name = "brelu")
  print brelu

  return brelu, dropoutDict, features, scalars, histograms