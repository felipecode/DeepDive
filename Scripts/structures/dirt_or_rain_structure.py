"""
Returns the last tensor of the network's structure.
Input is tensorflow class and an input placeholder.  
"""
def create_structure(tf, x):
 
  """Deep dive libs"""
  from deep_dive import DeepDive

  """Our little piece of network for ultimate underwater deconvolution and domination of the sea-world"""
  deep_dive = DeepDive()

  # """Creating weight variables"""
  # W_conv1 = deep_dive.weight_variable([16,16,3,512])
  # W_conv2 = deep_dive.weight_variable([1,1,512,512])
  # W_conv3 = deep_dive.weight_variable([8,8,512,3])

  """Creating weight variables"""
  W_conv1 = deep_dive.weight_variable_scaling([16,16,3,512], name='w_conv1')
  W_conv2 = deep_dive.weight_variable_scaling([1,1,512,512], name='w_conv2')
  W_conv3 = deep_dive.weight_variable_scaling([8,8,512,3], name='w_conv3') 

  """Creating bias variables"""
  b_conv1 = deep_dive.bias_variable([512])
  b_conv2 = deep_dive.bias_variable([512])
  b_conv3 = deep_dive.bias_variable([3])

  """Reshaping images"""
  x_image = tf.reshape(x, [-1,184,184,3], "unflattening_reshape")


  """Convolution Layers with sigmoids"""
  h_conv1 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1, padding='SAME') + b_conv1, name="first_sigmoid")
  h_conv2 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2, padding='SAME') + b_conv2, name="second_sigmoid")
  h_conv3 = tf.sigmoid(deep_dive.conv2d(h_conv2, W_conv3, padding='SAME') + b_conv3, name="third_sigmoid")

  return h_conv3