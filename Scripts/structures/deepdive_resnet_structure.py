"""
Returns the last tensor of the network's structure.
Input is tensorflow class and an input placeholder.  """
def create_structure(tf, x, input_size,dropout):
 
  """Deep dive libs"""
  from deep_dive import DeepDive

  """Our little piece of network for ultimate underwater deconvolution and domination of the sea-world"""
  deep_dive = DeepDive()

  """Reshaping images"""
  x_image = x
  print x
  dropoutDict={}
  features={}
  scalars={}



  """INCEPTION RES NET B"""



  W_resB_1 = deep_dive.weight_variable_scaling([1,1,3,24], name='W_resB_1')
  b_resB_1 = deep_dive.bias_variable([24])
  W_resB_2 = deep_dive.weight_variable_scaling([1,1,3,24], name='W_resB_2')
  b_resB_2 = deep_dive.bias_variable([24])
  W_resB_3 = deep_dive.weight_variable_scaling([1,3,24,28], name='W_resB_3')
  b_resB_3 = deep_dive.bias_variable([28])
  W_resB_4 = deep_dive.weight_variable_scaling([3,1,28,32], name='W_resB_4')
  b_resB_4 = deep_dive.bias_variable([32])
  W_resB_5 = deep_dive.weight_variable_scaling([1,1,60,3], name='W_resB_5')
  b_resB_5 = deep_dive.bias_variable([3])


  conv_resB_1=deep_dive.conv2d(x_image, W_resB_1,strides=[1, 1, 1, 1], padding='SAME') + b_resB_1
  conv_resB_2=deep_dive.conv2d(x_image, W_resB_2,strides=[1, 1, 1, 1], padding='SAME') + b_resB_2
  conv_resB_3=deep_dive.conv2d(conv_resB_2, W_resB_3,strides=[1, 1, 1, 1], padding='SAME') + b_resB_3
  conv_resB_4=deep_dive.conv2d(conv_resB_3, W_resB_4,strides=[1, 1, 1, 1], padding='SAME') + b_resB_4

  conv_resB_5_input=tf.concat(3, [conv_resB_3,conv_resB_4])

  conv_resB_5=deep_dive.conv2d(conv_resB_5_input, W_resB_5,strides=[1, 1, 1, 1], padding='SAME') + b_resB_5


  resB_output =  conv_resB_5 + x_image


  """INCEPTION RES NET A"""


  W_resA_1 = deep_dive.weight_variable_scaling([1,1,3,6], name='W_resA_1')
  b_resA_1 = deep_dive.bias_variable([6])
  W_resA_2 = deep_dive.weight_variable_scaling([1,1,3,6], name='W_resA_2')
  b_resA_2 = deep_dive.bias_variable([6])
  W_resA_3 = deep_dive.weight_variable_scaling([3,3,6,6], name='W_resA_3')
  b_resA_3 = deep_dive.bias_variable([6])
  W_resA_4 = deep_dive.weight_variable_scaling([1,1,3,8], name='W_resA_4')
  b_resA_4 = deep_dive.bias_variable([8])
  W_resA_5 = deep_dive.weight_variable_scaling([3,3,8,12], name='W_resA_5')
  b_resA_5 = deep_dive.bias_variable([12])
  W_resA_6 = deep_dive.weight_variable_scaling([3,3,12,16], name='W_resA_6')
  b_resA_6 = deep_dive.bias_variable([16])
  W_resA_7 = deep_dive.weight_variable_scaling([1,1,28,3], name='W_resA_7')
  b_resA_7 = deep_dive.bias_variable([3])


  conv_resA_1=deep_dive.conv2d(resB_output, W_resA_1,strides=[1, 1, 1, 1], padding='SAME') + b_resA_1
  conv_resA_2=deep_dive.conv2d(resB_output, W_resA_2,strides=[1, 1, 1, 1], padding='SAME') + b_resA_2
  conv_resA_3=deep_dive.conv2d(conv_resA_2, W_resA_3,strides=[1, 1, 1, 1], padding='SAME') + b_resA_3
  conv_resA_4=deep_dive.conv2d(resB_output, W_resA_4,strides=[1, 1, 1, 1], padding='SAME') + b_resA_4
  conv_resA_5=deep_dive.conv2d(conv_resA_4, W_resA_5,strides=[1, 1, 1, 1], padding='SAME') + b_resA_5
  conv_resA_6=deep_dive.conv2d(conv_resA_5, W_resA_6,strides=[1, 1, 1, 1], padding='SAME') + b_resA_6


  conv_resA_7_input=tf.concat(3, [conv_resA_1,conv_resA_3,conv_resA_6])

  conv_resA_7=deep_dive.conv2d(conv_resA_7_input, W_resA_7,strides=[1, 1, 1, 1], padding='SAME') + b_resA_7

  resA_ouput = conv_resA_7 + resB_output


  """INCEPTION RES NET C"""



  W_resC_1 = deep_dive.weight_variable_scaling([1,1,3,32], name='W_resC_1')
  b_resC_1 = deep_dive.bias_variable([32])
  W_resC_2 = deep_dive.weight_variable_scaling([1,1,3,32], name='W_resC_2')
  b_resC_2 = deep_dive.bias_variable([32])
  W_resC_3 = deep_dive.weight_variable_scaling([1,7,32,32], name='W_resC_3')
  b_resC_3 = deep_dive.bias_variable([32])
  W_resC_4 = deep_dive.weight_variable_scaling([7,1,32,32], name='W_resC_4')
  b_resC_4 = deep_dive.bias_variable([32])
  W_resC_5 = deep_dive.weight_variable_scaling([1,1,64,3], name='W_resC_5')
  b_resC_5 = deep_dive.bias_variable([3])


  conv_resC_1=deep_dive.conv2d(resA_ouput, W_resC_1,strides=[1, 1, 1, 1], padding='SAME') + b_resC_1
  conv_resC_2=deep_dive.conv2d(resA_ouput, W_resC_2,strides=[1, 1, 1, 1], padding='SAME') + b_resC_2
  conv_resC_3=deep_dive.conv2d(conv_resC_2, W_resC_3,strides=[1, 1, 1, 1], padding='SAME') + b_resC_3
  conv_resC_4=deep_dive.conv2d(conv_resC_3, W_resC_4,strides=[1, 1, 1, 1], padding='SAME') + b_resC_4

  conv_resC_5_input=tf.concat(3, [conv_resC_3,conv_resC_4])

  conv_resC_5=deep_dive.conv2d(conv_resC_5_input, W_resC_5,strides=[1, 1, 1, 1], padding='SAME') + b_resC_5


  resC_output = tf.nn.relu(conv_resC_5 + resA_ouput)

 
  return resC_output