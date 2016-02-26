"""
Returns the last tensor of the network's structure.
Input is tensorflow class and an input placeholder.  
"""
def create_structure(tf, x, input_size):
 
  """Deep dive libs"""
  from deep_dive import DeepDive

  """Our little piece of network for ultimate underwater deconvolution and domination of the sea-world"""
  deep_dive = DeepDive()

  # """Creating weight variables"""
  # W_conv1 = deep_dive.weight_variable([16,16,3,512])
  # W_conv2 = deep_dive.weight_variable([1,1,512,512])
  # W_conv3 = deep_dive.weight_variable([8,8,512,3])


  """Creating weight variables"""

  # W_conv1 = deep_dive.weight_variable_scaling([1,1,3,512], name='w_conv1')
  # W_conv2 = deep_dive.weight_variable_scaling([1,1,512,512], name='w_conv2')
  # W_conv3 = deep_dive.weight_variable_scaling([1,1,512,3], name='w_conv3') 



  # W_conv1 = deep_dive.weight_variable_scaling([1,1,3,128], name='w_conv1')
  # W_conv2 = deep_dive.weight_variable_scaling([1,1,128,128], name='w_conv2')
  # W_conv3 = deep_dive.weight_variable_scaling([1,1,128,128], name='w_conv3')
  # W_conv4 = deep_dive.weight_variable_scaling([1,1,128,3], name='w_conv4') 


  # W_conv1 = deep_dive.weight_variable_scaling([1,1,3,512], name='w_conv1')
  # W_conv2 = deep_dive.weight_variable_scaling([1,1,512,512], name='w_conv2')
  # W_conv3 = deep_dive.weight_variable_scaling([1,1,512,512], name='w_conv3')
  # W_conv4 = deep_dive.weight_variable_scaling([1,1,512,3], name='w_conv4') 



  # W_conv1 = deep_dive.weight_variable_scaling([32,32,3,128], name='w_conv1')
  # W_conv2 = deep_dive.weight_variable_scaling([1,1,128,128], name='w_conv2')
  # W_conv3 = deep_dive.weight_variable_scaling([32,32,128,3], name='w_conv3') 


  # W_conv1 = deep_dive.weight_variable_scaling([32,32,3,128], name='w_conv1')
  # # W_conv2 = deep_dive.weight_variable_scaling([1,1,128,128], name='w_conv2')
  # W_conv3 = deep_dive.weight_variable_scaling([32,32,128,3], name='w_conv3') 


  # W_conv1 = deep_dive.weight_variable_scaling([1,1,3,128], name='w_conv1')
  # W_conv2 = deep_dive.weight_variable_scaling([1,1,128,128], name='w_conv2')
  # W_conv3 = deep_dive.weight_variable_scaling([1,1,128,128], name='w_conv3')
  # W_conv4 = deep_dive.weight_variable_scaling([1,1,128,3], name='w_conv4') 


  # """Creating bias variables"""
  # # with tf.device('/gpu:1'):
  # b_conv1 = deep_dive.bias_variable([128])
  # b_conv2 = deep_dive.bias_variable([128])
  # b_conv3 = deep_dive.bias_variable([128])
  # b_conv4 = deep_dive.bias_variable([3])

  # """Reshaping images"""
  # # with tf.device('/gpu:2'):
  # x_image = tf.reshape(x, [-1,input_size[0],input_size[1],3], "unflattening_reshape")

  # """Create l2 regularizer"""
  # regularizer = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) + 
  #                tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
  #                tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
  #                tf.nn.l2_loss(W_conv4) + tf.nn.l2_loss(b_conv4))

  # """Convolution Layers with sigmoids"""
  # # with tf.device('/gpu:1'):
  # h_conv1 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1, padding='SAME') + b_conv1, name="first_sigmoid")
  # h_conv2 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2, padding='SAME') + b_conv2, name="second_sigmoid")
  # h_conv3 = tf.sigmoid(deep_dive.conv2d(h_conv2, W_conv3, padding='SAME') + b_conv3, name="third_sigmoid")
  # h_conv4 = deep_dive.conv2d(h_conv3, W_conv4, padding='SAME') + b_conv4

  # return h_conv4, regularizer

  """Inception test"""
  W_conv1_1_1 = deep_dive.weight_variable_scaling([1,1,3,32], name='w_conv1_1')
  W_conv1_3_3 = deep_dive.weight_variable_scaling([7,7,3,32], name='w_conv1_3')
  W_conv1_5_5 = deep_dive.weight_variable_scaling([15,15,3,32], name='w_conv1_5')

  W_conv2_1_1 = deep_dive.weight_variable_scaling([1,1,96,96], name='w_conv2_1')
  W_conv2_3_3 = deep_dive.weight_variable_scaling([3,3,96,96], name='w_conv2_3')
  W_conv2_5_5 = deep_dive.weight_variable_scaling([5,5,96,96], name='w_conv2_5')
  
  W_conv3_1_1 = deep_dive.weight_variable_scaling([1,1,96*3,3], name='w_conv3') 


  """Creating bias variables"""
  # with tf.device('/gpu:1'):
  b_conv1_1_1 = deep_dive.bias_variable([32])
  b_conv1_3_3 = deep_dive.bias_variable([32])
  b_conv1_5_5 = deep_dive.bias_variable([32])

  b_conv2_1_1 = deep_dive.bias_variable([96])
  b_conv2_3_3 = deep_dive.bias_variable([96])
  b_conv2_5_5 = deep_dive.bias_variable([96])

  b_conv3_1_1 = deep_dive.bias_variable([3])


  """Reshaping images"""
  # with tf.device('/gpu:2'):
  x_image = tf.reshape(x, [-1, input_size[0], input_size[1], 3], "unflattening_reshape")

  """Create l2 regularizer"""
  regularizer = (tf.nn.l2_loss(W_conv1_1_1) + tf.nn.l2_loss(b_conv1_1_1) + 
                 tf.nn.l2_loss(W_conv1_3_3) + tf.nn.l2_loss(b_conv1_3_3) +
                 tf.nn.l2_loss(W_conv1_5_5) + tf.nn.l2_loss(b_conv1_5_5) +
                 tf.nn.l2_loss(W_conv2_1_1) + tf.nn.l2_loss(b_conv2_1_1) +
                 tf.nn.l2_loss(W_conv2_3_3) + tf.nn.l2_loss(b_conv2_3_3) +
                 tf.nn.l2_loss(W_conv2_5_5) + tf.nn.l2_loss(b_conv2_5_5) +
                 tf.nn.l2_loss(W_conv3_1_1) + tf.nn.l2_loss(b_conv3_1_1))
  # regularizer = tf.constant(0.0)


  """Convolution Layers with sigmoids"""
  # with tf.device('/gpu:1'):
  h_conv1_1_1 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1_1_1, padding='SAME') + b_conv1_1_1, name="first_sigmoid")
  h_conv1_3_3 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1_3_3, padding='SAME') + b_conv1_3_3, name="second_sigmoid")
  h_conv1_5_5 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1_5_5, padding='SAME') + b_conv1_5_5, name="third_sigmoid")
  h_conv1 = tf.concat(3, [h_conv1_1_1, h_conv1_3_3, h_conv1_5_5])

  h_conv2_1_1 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2_1_1, padding='SAME') + b_conv2_1_1, name="first_sigmoid_2")
  h_conv2_3_3 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2_3_3, padding='SAME') + b_conv2_3_3, name="second_sigmoid_2")
  h_conv2_5_5 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2_5_5, padding='SAME') + b_conv2_5_5, name="third_sigmoid_2")
  h_conv2 = tf.concat(3, [h_conv2_1_1, h_conv2_3_3, h_conv2_5_5])

  h_conv3_1_1 = deep_dive.conv2d(h_conv2, W_conv3_1_1, padding='SAME') + b_conv3_1_1

  return h_conv3_1_1, regularizer

  # W1_1x1_3x3 = deep_dive.weight_variable_scaling([1,1,3,64], name='w1-1x1-3x3')
  # W1_1x1_5x5 = deep_dive.weight_variable_scaling([1,1,3,64], name='w1-1x1-5x5')
  # W1_1x1 = deep_dive.weight_variable_scaling([1,1,3,64], name='w1-1x1')
  # W1_3x3 = deep_dive.weight_variable_scaling([3,3,64,64], name='w1-3x3')
  # W1_5x5 = deep_dive.weight_variable_scaling([5,5,64,64], name='w1-5x5')

  # W2_1x1_3x3 = deep_dive.weight_variable_scaling([1,1,192,192], name='w2-1x1-3x3')
  # W2_1x1_5x5 = deep_dive.weight_variable_scaling([1,1,192,192], name='w2-1x1-5x5')
  # W2_1x1 = deep_dive.weight_variable_scaling([1,1,192,192], name='w2-1x1')
  # W2_3x3 = deep_dive.weight_variable_scaling([3,3,192,192], name='w2-3x3')
  # W2_5x5 = deep_dive.weight_variable_scaling([5,5,192,192], name='w2-5x5')

  # W3_1x1 = deep_dive.weight_variable_scaling([1,1,192*3,3], name='w3-1x1')

  # """Creating bias variables"""
  # # with tf.device('/gpu:1'):
  # b1_1x1_3x3 = deep_dive.bias_variable([64], name='b1-1x1-3x3')
  # b1_1x1_5x5 = deep_dive.bias_variable([64], name='b1-1x1-5x5')
  # b1_1x1 = deep_dive.bias_variable([64], name='b1-1x1')
  # b1_3x3 = deep_dive.bias_variable([64], name='b1-3x3')
  # b1_5x5 = deep_dive.bias_variable([64], name='b1-5x5')

  # b2_1x1_3x3 = deep_dive.bias_variable([192], name='b2-1x1-3x3')
  # b2_1x1_5x5 = deep_dive.bias_variable([192], name='b2-1x1-5x5')
  # b2_1x1 = deep_dive.bias_variable([192], name='b2-1x1')
  # b2_3x3 = deep_dive.bias_variable([192], name='b2-3x3')
  # b2_5x5 = deep_dive.bias_variable([192], name='b2-3x3')

  # b3_1x1 = deep_dive.bias_variable([3], name='b3-1x1')

  # """Reshaping images"""
  # # with tf.device('/gpu:2'):
  # x_image = tf.reshape(x, [-1, input_size[0], input_size[1], 3], "unflattening_reshape")

  # """Create l2 regularizer"""
  # # regularizer = (tf.nn.l2_loss(W_conv1_1_1) + tf.nn.l2_loss(b_conv1_1_1) + 
  # #                tf.nn.l2_loss(W_conv1_3_3) + tf.nn.l2_loss(b_conv1_3_3) +
  # #                tf.nn.l2_loss(W_conv1_5_5) + tf.nn.l2_loss(b_conv1_5_5) +
  # #                tf.nn.l2_loss(W_conv2_1_1) + tf.nn.l2_loss(b_conv2_1_1))
  # regularizer = tf.constant(0.0)

  # """Convolution Layers with sigmoids"""
  # # with tf.device('/gpu:1'):
  # h1_1x1_3x3 = tf.sigmoid(deep_dive.conv2d(x_image, W1_1x1_3x3, padding='SAME') + b1_1x1_3x3, name="h1-1x1-3x3")
  # h1_1x1_5x5 = tf.sigmoid(deep_dive.conv2d(x_image, W1_1x1_5x5, padding='SAME') + b1_1x1_5x5, name="h1-1x1-5x5")
  # h1_1x1 = tf.sigmoid(deep_dive.conv2d(x_image, W1_1x1, padding='SAME') + b1_1x1, name="h1-1x1")
  # h1_3x3 = tf.sigmoid(deep_dive.conv2d(h1_1x1_3x3, W1_3x3, padding='SAME') + b1_3x3, name="h1-3x3")
  # h1_5x5 = tf.sigmoid(deep_dive.conv2d(h1_1x1_5x5, W1_5x5, padding='SAME') + b1_5x5, name="h1-5x5")

  # h1 = tf.concat(3, [h1_1x1, h1_3x3, h1_5x5])

  # h2_1x1_3x3 = tf.sigmoid(deep_dive.conv2d(h1, W2_1x1_3x3, padding='SAME') + b2_1x1_3x3, name="h2-1x1-3x3")
  # h2_1x1_5x5 = tf.sigmoid(deep_dive.conv2d(h1, W2_1x1_5x5, padding='SAME') + b2_1x1_5x5, name="h2-1x1-5x5")
  # h2_1x1 = tf.sigmoid(deep_dive.conv2d(h1, W2_1x1, padding='SAME') + b2_1x1, name="h2-1x1")
  # h2_3x3 = tf.sigmoid(deep_dive.conv2d(h2_1x1_3x3, W2_3x3, padding='SAME') + b2_3x3, name="h2-3x3")
  # h2_5x5 = tf.sigmoid(deep_dive.conv2d(h2_1x1_5x5, W2_5x5, padding='SAME') + b2_5x5, name="h2-5x5")

  # h2 = tf.concat(3, [h2_1x1, h2_3x3, h2_5x5])

  # h3_1x1 = deep_dive.conv2d(h2, W3_1x1, padding='SAME') + b3_1x1

  # return h3_1x1, regularizer


  # """Different network for different channels using naive-inception"""
  # W1R_1x1 = deep_dive.weight_variable_scaling([1,1,1,32], name='W1R_1x1')
  # W1G_1x1 = deep_dive.weight_variable_scaling([1,1,1,32], name='W1G_1x1')
  # W1B_1x1 = deep_dive.weight_variable_scaling([1,1,1,32], name='W1B_1x1')

  # W1R_3x3 = deep_dive.weight_variable_scaling([3,3,1,32], name='W1R_3x3')
  # W1G_3x3 = deep_dive.weight_variable_scaling([3,3,1,32], name='W1G_3x3')
  # W1B_3x3 = deep_dive.weight_variable_scaling([3,3,1,32], name='W1B_3x3')

  # W1R_5x5 = deep_dive.weight_variable_scaling([5,5,1,32], name='W1R_5x5')
  # W1G_5x5 = deep_dive.weight_variable_scaling([5,5,1,32], name='W1G_5x5')
  # W1B_5x5 = deep_dive.weight_variable_scaling([5,5,1,32], name='W1B_5x5')

  # W2R_1x1 = deep_dive.weight_variable_scaling([1,1,96,96], name='W2R_1x1')
  # W2G_1x1 = deep_dive.weight_variable_scaling([1,1,96,96], name='W2G_1x1')
  # W2B_1x1 = deep_dive.weight_variable_scaling([1,1,96,96], name='W2B_1x1')

  # W3_1x1 = deep_dive.weight_variable_scaling([1,1,288,3], name='W3') 


  # """Creating bias variables"""
  # # with tf.device('/gpu:1'):
  # b1R_1x1 = deep_dive.bias_variable([32])
  # b1G_1x1 = deep_dive.bias_variable([32])
  # b1B_1x1 = deep_dive.bias_variable([32])

  # b1R_3x3 = deep_dive.bias_variable([32])
  # b1G_3x3 = deep_dive.bias_variable([32])
  # b1B_3x3 = deep_dive.bias_variable([32])

  # b1R_5x5 = deep_dive.bias_variable([32])
  # b1G_5x5 = deep_dive.bias_variable([32])
  # b1B_5x5 = deep_dive.bias_variable([32])

  # b2R_1x1 = deep_dive.bias_variable([96])
  # b2G_1x1 = deep_dive.bias_variable([96])
  # b2B_1x1 = deep_dive.bias_variable([96])

  # b3_1x1 = deep_dive.bias_variable([3])

  # """Reshaping images"""
  # # with tf.device('/gpu:2'):
  # x_image = tf.reshape(x, [-1, input_size[0], input_size[1], 3], "unflattening_reshape")

  # """Create l2 regularizer"""
  # # regularizer = (tf.nn.l2_loss(W_conv1_1_1) + tf.nn.l2_loss(b_conv1_1_1) + 
  # #                tf.nn.l2_loss(W_conv1_3_3) + tf.nn.l2_loss(b_conv1_3_3) +
  # #                tf.nn.l2_loss(W_conv1_5_5) + tf.nn.l2_loss(b_conv1_5_5) +
  # #                tf.nn.l2_loss(W_conv2_1_1) + tf.nn.l2_loss(b_conv2_1_1))
  # regularizer = tf.constant(0.0)

  # x_imageR = tf.reshape(x_image[:, :, :, 0], [-1, input_size[0], input_size[1], 1])
  # x_imageG = tf.reshape(x_image[:, :, :, 1], [-1, input_size[0], input_size[1], 1])
  # x_imageB = tf.reshape(x_image[:, :, :, 2], [-1, input_size[0], input_size[1], 1])

  # """Convolution Layers with sigmoids"""
  # # with tf.device('/gpu:1'):
  # h1R_1x1 = tf.sigmoid(deep_dive.conv2d(x_imageR, W1R_1x1, padding='SAME') + b1R_1x1, name="h1r")
  # h1G_1x1 = tf.sigmoid(deep_dive.conv2d(x_imageG, W1G_1x1, padding='SAME') + b1G_1x1, name="h1g")
  # h1B_1x1 = tf.sigmoid(deep_dive.conv2d(x_imageB, W1B_1x1, padding='SAME') + b1B_1x1, name="h1b")

  # h1R_3x3 = tf.sigmoid(deep_dive.conv2d(x_imageR, W1R_3x3, padding='SAME') + b1R_3x3, name="h1r_3x3")
  # h1G_3x3 = tf.sigmoid(deep_dive.conv2d(x_imageG, W1G_3x3, padding='SAME') + b1G_3x3, name="h1g_3x3")
  # h1B_3x3 = tf.sigmoid(deep_dive.conv2d(x_imageB, W1B_3x3, padding='SAME') + b1B_3x3, name="h1b_3x3")

  # h1R_5x5 = tf.sigmoid(deep_dive.conv2d(x_imageR, W1R_5x5, padding='SAME') + b1R_5x5, name="h1r_5x5")
  # h1G_5x5 = tf.sigmoid(deep_dive.conv2d(x_imageG, W1G_5x5, padding='SAME') + b1G_5x5, name="h1g_5x5")
  # h1B_5x5 = tf.sigmoid(deep_dive.conv2d(x_imageB, W1B_5x5, padding='SAME') + b1B_5x5, name="h1b_5x5")

  # h1R = tf.concat(3, [h1R_1x1, h1R_3x3, h1R_5x5])
  # h1G = tf.concat(3, [h1G_1x1, h1G_3x3, h1G_5x5])
  # h1B = tf.concat(3, [h1B_1x1, h1B_3x3, h1B_5x5])

  # h2R = tf.sigmoid(deep_dive.conv2d(h1R, W2R_1x1, padding='SAME') + b2R_1x1, name="h2r")
  # h2G = tf.sigmoid(deep_dive.conv2d(h1G, W2G_1x1, padding='SAME') + b2G_1x1, name="h2g")
  # h2B = tf.sigmoid(deep_dive.conv2d(h1B, W2B_1x1, padding='SAME') + b2B_1x1, name="h2b")

  # h2 = tf.concat(3, [h2R, h2G, h2B])

  # h3_1x1 = deep_dive.conv2d(h2, W3_1x1, padding='SAME') + b3_1x1

  # return h3_1x1, regularizer