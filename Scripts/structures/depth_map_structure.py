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
  W_S1_conv2 = deep_dive.weight_variable_scaling([3,3,64,512], name='w_conv2_1')
  b_S1_conv2 = deep_dive.bias_variable([512])
  S1_conv2 = tf.nn.relu(deep_dive.conv2d(S1_pool1, W_S1_conv2, padding='SAME') + b_S1_conv2, name="Scale1_second_relu")


  """ Max Pool 1 """
  S1_pool2 = tf.nn.max_pool(S1_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Scale1_second_Pool')

  print S1_pool2

  """Inception 1"""
  
  W_S1_incep1_1_1 = deep_dive.weight_variable_scaling([1,1,512,256], name='W_S1_incep1_1_1')
  W_S1_incep1_3_3r = deep_dive.weight_variable_scaling([1,1,512,128], name='W_S1_incep1_3_3r')
  W_S1_incep1_3_3 = deep_dive.weight_variable_scaling([3,3,128,256], name='W_S1_incep1_3_3')
  W_S1_incep1_5_5r = deep_dive.weight_variable_scaling([1,1,512,96], name='W_S1_incep1_5_5r')
  W_S1_incep1_5_5 = deep_dive.weight_variable_scaling([5,5,96,128], name='W_S1_incep1_5_5')


  b_S1_incep1_1_1 = deep_dive.bias_variable([256])
  b_S1_incep1_3_3r = deep_dive.bias_variable([128])
  b_S1_incep1_3_3 = deep_dive.bias_variable([256])
  b_S1_incep1_5_5r = deep_dive.bias_variable([96])
  b_S1_incep1_5_5 = deep_dive.bias_variable([128])



  S1_incep1_1_1 = tf.sigmoid(deep_dive.conv2d(S1_pool2, W_S1_incep1_1_1, padding='SAME') + b_S1_incep1_1_1, name="S1_incep1_1_1")
  S1_incep1_3_3r = tf.sigmoid(deep_dive.conv2d(S1_pool2, W_S1_incep1_3_3r, padding='SAME') + b_S1_incep1_3_3r, name="S1_incep1_3_3r")
  S1_incep1_3_3 = tf.sigmoid(deep_dive.conv2d(S1_incep1_3_3r, W_S1_incep1_3_3, padding='SAME') + b_S1_incep1_3_3, name="S1_incep1_3_3")
  S1_incep1_5_5r = tf.sigmoid(deep_dive.conv2d(S1_pool2, W_S1_incep1_5_5r, padding='SAME') + b_S1_incep1_5_5r, name="S1_incep1_5_5r")
  S1_incep1_5_5 = tf.sigmoid(deep_dive.conv2d(S1_incep1_5_5r, W_S1_incep1_5_5, padding='SAME') + b_S1_incep1_5_5, name="S1_incep1_5_5")


  S1_incep1 = tf.concat(3, [S1_incep1_1_1, S1_incep1_3_3, S1_incep1_5_5])


  print  S1_incep1



  """Inception 2"""


  W_S1_incep2_1_1 = deep_dive.weight_variable_scaling([1,1,640,512], name='W_S1_incep2_1_1')
  W_S1_incep2_3_3r = deep_dive.weight_variable_scaling([1,1,640,256], name='W_S1_incep2_3_3r')
  W_S1_incep2_3_3 = deep_dive.weight_variable_scaling([3,3,256,384], name='W_S1_incep2_3_3')
  W_S1_incep2_5_5r = deep_dive.weight_variable_scaling([1,1,640,128], name='W_S1_incep2_5_5r')
  W_S1_incep2_5_5 = deep_dive.weight_variable_scaling([5,5,128,192], name='W_S1_incep2_5_5')


  b_S1_incep2_1_1 = deep_dive.bias_variable([512])
  b_S1_incep2_3_3r = deep_dive.bias_variable([256])
  b_S1_incep2_3_3 = deep_dive.bias_variable([384])
  b_S1_incep2_5_5r = deep_dive.bias_variable([128])
  b_S1_incep2_5_5 = deep_dive.bias_variable([192])


 


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



  

  W_S1_up1 = deep_dive.weight_variable_scaling([1,1,1088,4096], name='w_up1_1')
  b_S1_up1 = deep_dive.bias_variable([4096])

  

  S1_pool3_up = tf.nn.relu(tf.nn.conv2d(S1_pool3, W_S1_up1 , strides=[1,1,1,1], padding='SAME', name=None) + b_S1_up1, name="Scale1_first_up")
  
  print S1_pool3_up

  S1_up1 = tf.depth_to_space(S1_pool3_up, 32 , name=None)

  print S1_up1
 
  #output_shape =tf.pack([batch_size,128,128,3])


  # There is a 16x16 to apply.
  W_S1_up2 = deep_dive.weight_variable_scaling([1,1,4,512], name='w_up2_1')
  b_S1_up2 = deep_dive.bias_variable([512])
  S1_up2 = tf.nn.relu(tf.nn.conv2d(S1_up1, W_S1_up2 , strides=[1,1,1,1], padding='SAME', name=None) + b_S1_up2, name="Scale1_second_up")
  
  print S1_up2

  S1_up2_final = tf.depth_to_space(S1_up2, 4 , name=None)
  print 'final'
  print S1_up2_final

  """Create l2 regularizer"""
  # regularizer = (tf.nn.l2_loss(W_conv1_1_1) + tf.nn.l2_loss(b_conv1_1_1) + 
  #                tf.nn.l2_loss(W_conv1_3_3) + tf.nn.l2_loss(b_conv1_3_3) +
  #                tf.nn.l2_loss(W_conv1_5_5) + tf.nn.l2_loss(b_conv1_5_5) +
  #                tf.nn.l2_loss(W_conv2_1_1) + tf.nn.l2_loss(b_conv2_1_1) +
  #                tf.nn.l2_loss(W_conv2_3_3) + tf.nn.l2_loss(b_conv2_3_3) +
  #                tf.nn.l2_loss(W_conv2_5_5) + tf.nn.l2_loss(b_conv2_5_5) +
  #                tf.nn.l2_loss(W_conv3_1_1) + tf.nn.l2_loss(b_conv3_1_1))
  regularizer = tf.constant(0.0)


 



  """ PART 2 SCALE 2 """




  """ Expand to 96 + 32 , Check this number """


  # Inception to be used directly is interesting.

  W_S2_conv1 = deep_dive.weight_variable_scaling([1,1,3,32], name='W_S2_conv1')
  b_S2_conv1 = deep_dive.bias_variable([32])
  S2_conv1 = tf.nn.relu(deep_dive.conv2d(x_image, W_S2_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_S2_conv1, name="Scale2_first_relu")

  print  S2_conv1

  S2_conv1 = tf.concat(3, [S2_conv1, S1_up2_final])


  print S2_conv1





  """ Inception 1 scale 2"""
 
  W_S2_incep1_1_1 = deep_dive.weight_variable_scaling([1,1,64,64], name='W_S2_incep1_1_1')
  W_S2_incep1_3_3 = deep_dive.weight_variable_scaling([7,7,64,32], name='W_S2_incep1_3_3')
  W_S2_incep1_5_5 = deep_dive.weight_variable_scaling([15,15,64,32], name='W_S2_incep1_5_5')


  b_S2_incep1_1_1 = deep_dive.bias_variable([64])
  b_S2_incep1_3_3 = deep_dive.bias_variable([32])
  b_S2_incep1_5_5 = deep_dive.bias_variable([32])



  S2_incep1_1_1 = tf.sigmoid(deep_dive.conv2d(S2_conv1, W_S2_incep1_1_1, padding='SAME') + b_S2_incep1_1_1, name="S2_incep1_1_1")
  S2_incep1_3_3 = tf.sigmoid(deep_dive.conv2d(S2_conv1, W_S2_incep1_3_3, padding='SAME') + b_S2_incep1_3_3, name="S2_incep1_3_3")
  S2_incep1_5_5 = tf.sigmoid(deep_dive.conv2d(S2_conv1, W_S2_incep1_5_5, padding='SAME') + b_S2_incep1_5_5, name="S2_incep1_5_5")


  S2_incep1 = tf.concat(3, [S2_incep1_1_1, S2_incep1_3_3, S2_incep1_5_5])


  print  S2_incep1



  """Inception 2 scale 1"""


  W_S2_incep2_1_1 = deep_dive.weight_variable_scaling([1,1,128,96], name='W_S2_incep2_1_1')
  W_S2_incep2_3_3 = deep_dive.weight_variable_scaling([3,3,128,96], name='W_S2_incep2_3_3')
  W_S2_incep2_5_5 = deep_dive.weight_variable_scaling([5,5,128,64], name='W_S2_incep2_5_5')


  b_S2_incep2_1_1 = deep_dive.bias_variable([96])
  b_S2_incep2_3_3 = deep_dive.bias_variable([96])
  b_S2_incep2_5_5 = deep_dive.bias_variable([64])



  S2_incep2_1_1 = tf.sigmoid(deep_dive.conv2d(S2_incep1, W_S2_incep2_1_1, padding='SAME') + b_S2_incep2_1_1, name="S2_incep2_1_1")
  S2_incep2_3_3 = tf.sigmoid(deep_dive.conv2d(S2_incep1, W_S2_incep2_3_3, padding='SAME') + b_S2_incep2_3_3, name="S2_incep2_3_3")
  S2_incep2_5_5 = tf.sigmoid(deep_dive.conv2d(S2_incep1, W_S2_incep2_5_5, padding='SAME') + b_S2_incep2_5_5, name="S2_incep2_5_5")


  S2_incep1 = tf.concat(3, [S2_incep2_1_1, S2_incep2_3_3, S2_incep2_5_5])



  print  S2_incep1





  W_S2_conv2 = deep_dive.weight_variable_scaling([1,1,256,3], name='W_S2_conv2')
  b_S2_conv2 = deep_dive.bias_variable([3])
  S2_conv2 = tf.nn.relu(deep_dive.conv2d(S2_incep1, W_S2_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_S2_conv2, name="Scale2_second_relu")






  print S2_conv2

  return S2_conv2,regularizer





 # W_conv1_1_1 = deep_dive.weight_variable_scaling([1,1,3,32], name='w_conv1_1')
 #  W_conv1_3_3 = deep_dive.weight_variable_scaling([7,7,3,32], name='w_conv1_3')
 #  W_conv1_5_5 = deep_dive.weight_variable_scaling([15,15,3,32], name='w_conv1_5')

 #  W_conv2_1_1 = deep_dive.weight_variable_scaling([1,1,96,96], name='w_conv2_1')
 #  W_conv2_3_3 = deep_dive.weight_variable_scaling([3,3,96,96], name='w_conv2_3')
 #  W_conv2_5_5 = deep_dive.weight_variable_scaling([5,5,96,96], name='w_conv2_5')
  
 #  W_conv3_1_1 = deep_dive.weight_variable_scaling([1,1,96*3,3], name='w_conv3') 


 #  """Creating bias variables"""
 #  # with tf.device('/gpu:1'):
 #  b_conv1_1_1 = deep_dive.bias_variable([32])
 #  b_conv1_3_3 = deep_dive.bias_variable([32])
 #  b_conv1_5_5 = deep_dive.bias_variable([32])

 #  b_conv2_1_1 = deep_dive.bias_variable([96])
 #  b_conv2_3_3 = deep_dive.bias_variable([96])
 #  b_conv2_5_5 = deep_dive.bias_variable([96])

 #  b_conv3_1_1 = deep_dive.bias_variable([3])


 #  """Reshaping images"""
 #  # with tf.device('/gpu:2'):
 #  x_image = tf.reshape(x, [-1, input_size[0], input_size[1], 3], "unflattening_reshape")

 #  """Create l2 regularizer"""
 #  regularizer = (tf.nn.l2_loss(W_conv1_1_1) + tf.nn.l2_loss(b_conv1_1_1) + 
 #                 tf.nn.l2_loss(W_conv1_3_3) + tf.nn.l2_loss(b_conv1_3_3) +
 #                 tf.nn.l2_loss(W_conv1_5_5) + tf.nn.l2_loss(b_conv1_5_5) +
 #                 tf.nn.l2_loss(W_conv2_1_1) + tf.nn.l2_loss(b_conv2_1_1) +
 #                 tf.nn.l2_loss(W_conv2_3_3) + tf.nn.l2_loss(b_conv2_3_3) +
 #                 tf.nn.l2_loss(W_conv2_5_5) + tf.nn.l2_loss(b_conv2_5_5) +
 #                 tf.nn.l2_loss(W_conv3_1_1) + tf.nn.l2_loss(b_conv3_1_1))
 #  # regularizer = tf.constant(0.0)


 #  """Convolution Layers with sigmoids"""
 #  # with tf.device('/gpu:1'):
 #  h_conv1_1_1 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1_1_1, padding='SAME') + b_conv1_1_1, name="first_sigmoid")
 #  h_conv1_3_3 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1_3_3, padding='SAME') + b_conv1_3_3, name="second_sigmoid")
 #  h_conv1_5_5 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1_5_5, padding='SAME') + b_conv1_5_5, name="third_sigmoid")
 #  h_conv1 = tf.concat(3, [h_conv1_1_1, h_conv1_3_3, h_conv1_5_5])

 #  h_conv2_1_1 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2_1_1, padding='SAME') + b_conv2_1_1, name="first_sigmoid_2")
 #  h_conv2_3_3 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2_3_3, padding='SAME') + b_conv2_3_3, name="second_sigmoid_2")
 #  h_conv2_5_5 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2_5_5, padding='SAME') + b_conv2_5_5, name="third_sigmoid_2")
 #  h_conv2 = tf.concat(3, [h_conv2_1_1, h_conv2_3_3, h_conv2_5_5])

 #  h_conv3_1_1 = deep_dive.conv2d(h_conv2, W_conv3_1_1, padding='SAME') + b_conv3_1_1

