def create_structure(tf, x):
 
  """Deep dive libs"""
  from deep_dive import DeepDive
  import input_data_dive

  # global_step = tf.Variable(0, trainable=False, name="global_step")

  # Our little piece of network for ultimate underwater deconvolution and domination of the sea-world

  deep_dive = DeepDive()
  path = '../Local_aux/weights/'

  W_conv1 = deep_dive.weight_variable([9,9,3,64])

  #W_smooth = deep_dive.weight_variable([1, 1, 38, 1])

  W_conv2 = deep_dive.weight_variable([1,1,64,32])

  W_conv3 = deep_dive.weight_variable([5,5,32,3])

  b_conv1 = deep_dive.bias_variable([64])

  b_conv2 = deep_dive.bias_variable([32])

  b_conv3 = deep_dive.bias_variable([3])

  #x_image = tf.reshape(x, [-1,184,184,3])


  x_image = tf.reshape(x, [-1,1200,815,3], "unflattening_reshape")

  """Red Channel"""
  # x_imageR =  tf.reshape(xR, [-1,184,184,1])
  h_conv1 = tf.nn.relu(deep_dive.conv2d(x_image, W_conv1, padding='SAME') + b_conv1, name="first_sigmoid")

  # h_conv1 = deep_dive.dropout(h_conv1)

  h_conv2 = tf.nn.relu(deep_dive.conv2d(h_conv1, W_conv2, padding='SAME') + b_conv2, name="second_sigmoid")
  # visualize(tf,h_conv1,W_conv2,b_conv2)

  # h_conv2 = deep_dive.dropout(h_conv2)

  # h_conv2 = deep_dive.dropout(h_conv2)

  h_conv3 = deep_dive.conv2d(h_conv2, W_conv3, padding='SAME') + b_conv3

  # h_conv3 = tf.nn.l2_normalize(h_conv3, 2)

  return h_conv3