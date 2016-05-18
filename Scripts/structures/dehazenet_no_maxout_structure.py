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
  histograms={}
  """Feature Extraction"""
  #conv1
  # INPUT IS ONE PATCH 3x16x16 Color
  # applies 5x5 filters with 0 padding 16 times
  # OUTPUT 16 X 12 X 12
  W_conv1 = deep_dive.weight_variable_scaling([5,5,3,16], name='W_conv1')
  b_conv1 = deep_dive.bias_variable([16])

  histograms["W_conv1"]=W_conv1

  histograms["b_conv1"]=b_conv1

  """print  conv1"""

  conv1=deep_dive.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='VALID') + b_conv1
  features["conv1"]=conv1

  #conv2 to replace maxout
  # INPUT IS ONE PATCH 16x12x12
  # applies 3x3 filters with 1 padding
  #apply reduction between the features, reducing from 16 to 4
  W_conv2_1 = deep_dive.weight_variable_scaling([3,3,16,4], name='W_conv2_1')
  b_conv2_1 = deep_dive.bias_variable([4])
  histograms["W_conv2_1"]=W_conv2_1
  histograms["b_conv2_1"]=b_conv2_1

  conv2_1=deep_dive.conv2d(conv1, W_conv2_1,strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1
  features["conv2_1"]=conv2_1
  
  """Inception 1 """
  """ TODO: , padding with variable size """

  """Multi-scale Mapping"""
  #inception
  # INPUT ARE PATCHES 4x12x12
  # applies convolution with 3 kernels with different dimensions
  # OUTPUT 48x12x12


  W_incep1_3_3 = deep_dive.weight_variable_scaling([3,3,4,16], name='W_incep1_3_3')
  W_incep1_5_5 = deep_dive.weight_variable_scaling([5,5,4,16], name='W_incep1_5_5')
  W_incep1_7_7 = deep_dive.weight_variable_scaling([7,7,4,16], name='W_incep1_7_7')

  b_incep1_3_3 = deep_dive.bias_variable([16])
  b_incep1_5_5 = deep_dive.bias_variable([16])
  b_incep1_7_7 = deep_dive.bias_variable([16])
  histograms["W_incep1_3_3"]=W_incep1_3_3
  histograms["W_incep1_5_5"]=W_incep1_5_5
  histograms["W_incep1_7_7"]=W_incep1_7_7
  histograms["b_incep1_3_3"]=b_incep1_3_3
  histograms["b_incep1_5_5"]=b_incep1_5_5
  histograms["b_incep1_7_7"]=b_incep1_7_7

  incep1_3_3 = deep_dive.conv2d(conv2_1, W_incep1_3_3, padding='SAME') + b_incep1_3_3
  incep1_5_5 = deep_dive.conv2d(conv2_1, W_incep1_5_5, padding='SAME') + b_incep1_5_5
  incep1_7_7 = deep_dive.conv2d(conv2_1, W_incep1_7_7, padding='SAME') + b_incep1_7_7
  features["incep1_3_3"]=incep1_3_3
  features["incep1_5_5"]=incep1_5_5
  features["incep1_7_7"]=incep1_7_7
  incep1 = tf.concat(3, [incep1_3_3, incep1_5_5, incep1_7_7])

  print  incep1

  """Local Extremum"""
  #MAXPOOL
  # INPUT 48x12x12
  # KERNEL 7x7
  # OUTPUT 48x6x6
  pool2 = tf.nn.max_pool(incep1, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='VALID', name='second_Pool')

  print pool2
  features["pool2"]=pool2
  W_conv2 = deep_dive.weight_variable_scaling([4,4,48,1], name='W_conv2')
  b_conv2 = deep_dive.bias_variable([1])
  histograms["W_conv2"]=W_conv2
  histograms["b_conv2"]=b_conv2
  one_constant = tf.constant(1)
  brelu = tf.minimum(tf.to_float(one_constant),tf.nn.relu(deep_dive.conv2d(pool2, W_conv2, padding='SAME') + b_conv2, name="second_relu"),name='brelu')
  print brelu
 
  return brelu,dropoutDict,features,scalars,histograms