"""
Returns the last tensor of the network's structure.
Input is tensorflow class and an input placeholder.  
"""
def create_structure(tf, x, input_size):
 
  """Deep dive libs"""
  from deep_dive import DeepDive

  """Our little piece of network for ultimate underwater deconvolution and domination of the sea-world"""
  deep_dive = DeepDive()



  """Reshaping images"""
  # with tf.device('/gpu:2'):
  x_image = tf.reshape(x, [-1, input_size[0], input_size[1], 3], "unflattening_reshape")


  """ Scale 1 """



  """Conv 1 """
  W_S1_conv1 = deep_dive.weight_variable_scaling([7,7,3,64], name='W_S1_conv1')
  b_S1_conv1 = deep_dive.bias_variable([64])
  S1_conv1 = tf.nn.relu(deep_dive.conv2d(x_image, W_S1_conv1,strides=[1, 2, 2, 1], padding='SAME') + b_S1_conv1, name="Scale1_first_relu")

  """ Max Pool 1 """

  # Belive this will take the max of a windown 2x2 with stride 1
  S1_pool1 = tf.nn.max_pool(S1_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Scale1_first_Pool')

  print S1_pool1

  """Conv 2 """
  W_S1_conv2 = deep_dive.weight_variable_scaling([3,3,64,192], name='w_conv2_1')
  b_S1_conv2 = deep_dive.bias_variable([192])
  S1_conv2 = tf.nn.relu(deep_dive.conv2d(S1_pool1, W_S1_conv2, padding='SAME') + b_S1_conv2, name="Scale1_second_relu")


  """ Max Pool 1 """
  S1_pool2 = tf.nn.max_pool(S1_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Scale1_second_Pool')

  print S1_pool2

  """Inception 1"""
  
  W_S1_incep1_1_1 = deep_dive.weight_variable_scaling([1,1,192,96], name='W_S1_incep1_1_1')
  W_S1_incep1_3_3r = deep_dive.weight_variable_scaling([1,1,192,96], name='W_S1_incep1_3_3r')
  W_S1_incep1_3_3 = deep_dive.weight_variable_scaling([3,3,96,128], name='W_S1_incep1_3_3')
  W_S1_incep1_5_5r = deep_dive.weight_variable_scaling([1,1,192,16], name='W_S1_incep1_5_5r')
  W_S1_incep1_5_5 = deep_dive.weight_variable_scaling([5,5,16,32], name='W_S1_incep1_5_5')


  b_S1_incep1_1_1 = deep_dive.bias_variable([96])
  b_S1_incep1_3_3r = deep_dive.bias_variable([96])
  b_S1_incep1_3_3 = deep_dive.bias_variable([128])
  b_S1_incep1_5_5r = deep_dive.bias_variable([16])
  b_S1_incep1_5_5 = deep_dive.bias_variable([32])



  S1_incep1_1_1 = tf.sigmoid(deep_dive.conv2d(S1_pool2, W_S1_incep1_1_1, padding='SAME') + b_S1_incep1_1_1, name="S1_incep1_1_1")
  S1_incep1_3_3r = tf.sigmoid(deep_dive.conv2d(S1_pool2, W_S1_incep1_3_3r, padding='SAME') + b_S1_incep1_3_3r, name="S1_incep1_3_3r")
  S1_incep1_3_3 = tf.sigmoid(deep_dive.conv2d(S1_incep1_3_3r, W_S1_incep1_3_3, padding='SAME') + b_S1_incep1_3_3, name="S1_incep1_3_3")
  S1_incep1_5_5r = tf.sigmoid(deep_dive.conv2d(S1_pool2, W_S1_incep1_5_5r, padding='SAME') + b_S1_incep1_5_5r, name="S1_incep1_5_5r")
  S1_incep1_5_5 = tf.sigmoid(deep_dive.conv2d(S1_incep1_5_5r, W_S1_incep1_5_5, padding='SAME') + b_S1_incep1_5_5, name="S1_incep1_5_5")


  S1_incep1 = tf.concat(3, [S1_incep1_1_1, S1_incep1_3_3, S1_incep1_5_5])


  print  S1_incep1



  """Inception 2"""


  W_S1_incep2_1_1 = deep_dive.weight_variable_scaling([1,1,256,192], name='W_S1_incep2_1_1')
  W_S1_incep2_3_3r = deep_dive.weight_variable_scaling([1,1,256,128], name='W_S1_incep2_3_3r')
  W_S1_incep2_3_3 = deep_dive.weight_variable_scaling([3,3,128,192], name='W_S1_incep2_3_3')
  W_S1_incep2_5_5r = deep_dive.weight_variable_scaling([1,1,256,32], name='W_S1_incep2_5_5r')
  W_S1_incep2_5_5 = deep_dive.weight_variable_scaling([5,5,32,96], name='W_S1_incep2_5_5')


  b_S1_incep2_1_1 = deep_dive.bias_variable([192])
  b_S1_incep2_3_3r = deep_dive.bias_variable([128])
  b_S1_incep2_3_3 = deep_dive.bias_variable([192])
  b_S1_incep2_5_5r = deep_dive.bias_variable([32])
  b_S1_incep2_5_5 = deep_dive.bias_variable([96])


 


  S1_incep2_1_1 = tf.sigmoid(deep_dive.conv2d(S1_incep1, W_S1_incep2_1_1, padding='SAME') + b_S1_incep2_1_1, name="S1_incep2_1_1")
  S1_incep2_3_3r = tf.sigmoid(deep_dive.conv2d(S1_incep1, W_S1_incep2_3_3r, padding='SAME') + b_S1_incep2_3_3r, name="S1_incep2_3_3r")
  S1_incep2_3_3 = tf.sigmoid(deep_dive.conv2d(S1_incep2_3_3r, W_S1_incep2_3_3, padding='SAME') + b_S1_incep2_3_3, name="S1_incep2_3_3")
  S1_incep2_5_5r = tf.sigmoid(deep_dive.conv2d(S1_incep1, W_S1_incep2_5_5r, padding='SAME') + b_S1_incep2_5_5r, name="S1_incep2_5_5r")
  S1_incep2_5_5 = tf.sigmoid(deep_dive.conv2d(S1_incep2_5_5r, W_S1_incep2_5_5, padding='SAME') + b_S1_incep2_5_5, name="S1_incep2_5_5")


  S1_incep2 = tf.concat(3, [S1_incep2_1_1, S1_incep2_3_3, S1_incep2_5_5])



  print  S1_incep2

  """ Average Polling just to finalize """


  S1_pool3 = tf.nn.avg_pool(S1_incep2, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME', name='Scale1_last_Pool')

  print S1_pool3

  """ Upsampling 1 to go back to original size """

  # There is a 16x16 to apply.
  #print x_image
  #batch_size = tf.shape(x_image)[0]
  #output_shape =tf.pack([batch_size,16,16,256])
  #print batch_size
  #print output_shape



  S1_pool3_up = tf.depth_to_space(S1_pool3, 2 , name=None)

  W_S1_up1 = deep_dive.weight_variable_scaling([16,16,240,240], name='w_up1_1')
  b_S1_up1 = deep_dive.bias_variable([240])


  S1_up1 = tf.nn.relu(tf.nn.conv2d(S1_pool3_up, W_S1_up1 , strides=[1,1,1,1], padding='SAME', name=None) + b_S1_up1, name="Scale1_first_up")
  
  

  print S1_up1
  """ Upsampling 2 to go back to original size """
  output_shape =tf.pack([batch_size,128,128,3])


  # There is a 16x16 to apply.
  W_S1_up2 = deep_dive.weight_variable_scaling([16,16,3,256], name='w_up2_1')
  b_S1_up2 = deep_dive.bias_variable([3])
  S1_up2 = tf.nn.relu(tf.nn.conv2d_transpose(S1_up1, W_S1_up2, output_shape , strides=[1,1,1,1], padding='SAME', name=None) + b_S1_up2, name="Scale1_second_up")
  


  """Create l2 regularizer"""
  # regularizer = (tf.nn.l2_loss(W_conv1_1_1) + tf.nn.l2_loss(b_conv1_1_1) + 
  #                tf.nn.l2_loss(W_conv1_3_3) + tf.nn.l2_loss(b_conv1_3_3) +
  #                tf.nn.l2_loss(W_conv1_5_5) + tf.nn.l2_loss(b_conv1_5_5) +
  #                tf.nn.l2_loss(W_conv2_1_1) + tf.nn.l2_loss(b_conv2_1_1) +
  #                tf.nn.l2_loss(W_conv2_3_3) + tf.nn.l2_loss(b_conv2_3_3) +
  #                tf.nn.l2_loss(W_conv2_5_5) + tf.nn.l2_loss(b_conv2_5_5) +
  #                tf.nn.l2_loss(W_conv3_1_1) + tf.nn.l2_loss(b_conv3_1_1))
  # regularizer = tf.constant(0.0)

  print S1_up2
 
  return S1_up2,S1_up2

