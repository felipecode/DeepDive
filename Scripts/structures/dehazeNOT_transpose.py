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
  	W_conv1 = deep_dive.weight_variable_scaling([3, 3, 3, 12], name = 'W_conv1')
  	b_conv1 = deep_dive.bias_variable([12])

  	histograms["W_conv1"] = W_conv1
  	histograms["b_conv1"] = b_conv1

  	conv1 = deep_dive.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv1
  	features["conv1"] = conv1

    #second convolution
    #INPUT: 14x14x12
    #KERNEL: 5x5
    #PADDING: 0
    #TIMES APPLIED: 16
    #OUTPUT: 10x10x16

    W_conv2 = deep_dive.weight_variable_scaling([5, 5, 12, 16], name = 'W_conv2')
    b_conv2 = deep_dive.bias_variable([16])

    histograms["W_conv2"] = W_conv2
    histograms["b_conv2"] = b_conv2

    conv2 = deep_dive.conv2d(conv1, W_conv2, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv2
    features["conv2"] = conv2

    #third convolution
    #INPUT: 10x10x16
    #KERNEL: 5x5
    #PADDING: 0
    #TIMES APPLIED: 32
    #OUTPUT: 6x6x32

    W_conv3 = deep_dive.weight_variable_scaling([5, 5, 16, 32], name = 'W_conv3')
    b_conv3 = deep_dive.bias_variable([32])

    histograms["W_conv3"] = W_conv3
    histograms["b_conv3"] = b_conv3

    conv3 = deep_dive.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv3
    features["conv3"] = conv3

    #fourth convolution
    #INPUT: 6x6x32
    #KERNEL: 5x5
    #PADDING: 0
    #TIMES APPLIED: 64
    #OUTPUT: 2x2x64

    W_conv4 = deep_dive.weight_variable_scaling([5, 5, 32, 64], name = 'W_conv4')
    b_conv4 = deep_dive.bias_variable([64])

    histograms["W_conv4"] = W_conv4
    histograms["b_conv4"] = b_conv4

    conv4 = deep_dive.conv2d(conv3, W_conv4, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv4
    features["conv4"] = conv4

    #first transpose convolution
    #INPUT: 2x2x64
    #KERNEL: 5x5
    #PADDING: 0
    #TIMES APPLIED: 32
    #OUTPUT: 6x6x32

    W_deconv1 = deep_dive.weight_variable_scaling([5, 5, 64, 32], name = 'W_deconv1')
    b_deconv1 = deep_dive.bias_variable([32])

    histograms["W_deconv1"] = W_deconv1
    histograms["b_deconv1"] = b_deconv1

    deconv1 = deep_dive.conv2d_transpose(conv4, W_deconv1, output = [1, 6, 6, 32], strides = [1, 1, 1, 1], padding = 'VALID') + b_deconv1
    features["deconv1"] = deconv1

    #second transpose convolution
    #INPUT: 6x6x32
    #KERNEL: 5x5
    #PADDING: 0
    #TIMES APPLIED: 16
    #OUTPUT: 10x10x16

    W_deconv2 = deep_dive.weight_variable_scaling([5, 5, 32, 16], name = 'W_deconv2')
    b_deconv2 = deep_dive.bias_variable([16])

    histograms["W_deconv2"] = W_deconv2
    histograms["b_deconv2"] = b_deconv2

    deconv2 = deep_dive.conv2d_transpose(deconv1, W_deconv2, output = [1, 10, 10, 16], strides = [1, 1, 1, 1], padding = 'VALID') + b_deconv2
    features["deconv2"] = deconv2

    #third transpose convolution
    #INPUT: 10x10x16
    #KERNEL: 5x5
    #PADDING: 0
    #TIMES APPLIED: 32
    #OUTPUT: 14x14x12

    W_deconv3 = deep_dive.weight_variable_scaling([5, 5, 16, 12], name = 'W_deconv3')
    b_deconv3 = deep_dive.bias_variable([12])

    histograms["W_deconv3"] = W_deconv3
    histograms["b_deconv3"] = b_deconv3

    deconv3 = deep_dive.conv2d_transpose(deconv2, W_deconv3, output = [1, 14, 14, 12], strides = [1, 1, 1, 1], padding = 'VALID') + b_deconv3
    features["deconv3"] = deconv3

    #fourth transpose convolution
    #INPUT: 14x14x12
    #KERNEL: 3x3
    #PADDING: 0
    #TIMES APPLIED: 1
    #OUTPUT: 16x16x1

    W_deconv4 = deep_dive.weight_variable_scaling([3, 3, 12, 1], name = 'W_deconv4')
    b_deconv4 = deep_dive.bias_variable([1])

    histograms["W_deconv4"] = W_deconv4
    histograms["b_deconv4"] = b_deconv4

    deconv4 = deep_dive.conv2d_transpose(deconv3, W_deconv4, output = [1, 16, 16, 1], strides = [1, 1, 1, 1], padding = 'VALID') + b_deconv4
    features["deconv4"] = deconv4

    one_constant = tf.constant(1)

    brelu = tf.minimum(tf.to_float(one_constant), tf.nn.relu(deconv4, name = "relu"), name = "brelu")
    print brelu

    return brelu, dropoutDict, features, scalars, histograms