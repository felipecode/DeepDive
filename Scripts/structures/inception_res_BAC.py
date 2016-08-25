"""
Returns the last tensor of the network's structure
followed dropout and features dictionaries to be summarised
Input is tensorflow class and an input placeholder.  
"""

def create_structure(tf, x, input_size,dropout):
 
  """Deep dive libs"""
  from deep_dive import DeepDive
  deep_dive = DeepDive()



  dropoutDict={}

  x_image =x
  """ Scale 1 """

  features={}
  scalars={}
  histograms={}

  W_conv1 = deep_dive.weight_variable_scaling([3,3,3,16],name='W_conv1')
  b_conv1 = deep_dive.bias_variable([16])
  conv1 = deep_dive.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1
  features["conv1"]=[conv1, W_conv1]



  W_B_conv1 = deep_dive.weight_variable_scaling([1,1,16,24], name='W_B_conv1')
  b_B_conv1 = deep_dive.bias_variable([24])
  B_conv1 = deep_dive.conv2d(conv1, W_B_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv1
  print B_conv1
  features["B_conv1"]=[B_conv1,None]
  histograms["W_B_conv1"]=W_B_conv1
  histograms["b_B_conv1"]=b_B_conv1


  W_B_conv2 = deep_dive.weight_variable_scaling([1,1,16,24], name='W_B_conv2')
  b_B_conv2 = deep_dive.bias_variable([24])
  B_conv2 = deep_dive.conv2d(conv1, W_B_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv2
  print B_conv2
  features["B_conv2"]=[B_conv2,None]
  histograms["W_B_conv2"]=W_B_conv2
  histograms["b_B_conv2"]=b_B_conv2

  W_B_conv3 = deep_dive.weight_variable_scaling([1,3,24,28], name='W_B_conv3')
  b_B_conv3 = deep_dive.bias_variable([28])
  B_conv3 = deep_dive.conv2d(B_conv2, W_B_conv3,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv3
  print B_conv3
  features["B_conv3"]=[B_conv3,None]
  histograms["W_B_conv3"]=W_B_conv3
  histograms["b_B_conv3"]=b_B_conv3

  
  W_B_conv4 = deep_dive.weight_variable_scaling([3,1,28,32], name='W_B_conv4')
  b_B_conv4 = deep_dive.bias_variable([32])
  B_conv4 = deep_dive.conv2d(B_conv3, W_B_conv4,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv4
  print B_conv4
  features["B_conv4"]=[B_conv4,None]
  histograms["W_B_conv4"]=W_B_conv4
  histograms["b_B_conv4"]=b_B_conv4

  B_concat = tf.concat(3, [B_conv1,B_conv4])

  W_B_conv5 = deep_dive.weight_variable_scaling([1,1,56,16], name='W_B_conv5')
  b_B_conv5 = deep_dive.bias_variable([16])
  B_conv5 = deep_dive.conv2d(B_concat, W_B_conv5,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv5
  print B_conv5
  features["B_conv5"]=[B_conv5,None]
  histograms["W_B_conv5"]=W_B_conv5
  histograms["b_B_conv5"]=b_B_conv5

  B_relu = tf.nn.relu(B_conv5+ conv1)
  features["B_relu"]=[B_relu,None]

  W_A_conv1 = deep_dive.weight_variable_scaling([1,1,16,6], name='W_A_conv1')
  b_A_conv1 = deep_dive.bias_variable([6])
  A_conv1 = deep_dive.conv2d(B_relu, W_A_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv1
  print A_conv1
  features["A_conv1"]=[A_conv1,None]
  histograms["W_A_conv1"]=W_A_conv1
  histograms["b_A_conv1"]=b_A_conv1

  W_A_conv2 = deep_dive.weight_variable_scaling([1,1,16,6], name='W_A_conv2')
  b_A_conv2 = deep_dive.bias_variable([6])
  A_conv2 = deep_dive.conv2d(B_relu, W_A_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv2
  print A_conv2
  features["A_conv2"]=[A_conv2,None]
  histograms["W_A_conv2"]=W_A_conv2
  histograms["b_A_conv2"]=b_A_conv2

  W_A_conv3 = deep_dive.weight_variable_scaling([3,3,6,6], name='W_A_conv3')
  b_A_conv3 = deep_dive.bias_variable([6])
  A_conv3 = deep_dive.conv2d(A_conv2, W_A_conv3,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv3
  print A_conv3
  features["A_conv3"]=[A_conv3,None]
  histograms["W_A_conv3"]=W_A_conv3
  histograms["b_A_conv3"]=b_A_conv3

  W_A_conv4 = deep_dive.weight_variable_scaling([1,1,16,8], name='W_A_conv4')
  b_A_conv4 = deep_dive.bias_variable([8])
  A_conv4 = deep_dive.conv2d(B_relu, W_A_conv4,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv4
  print A_conv4
  features["A_conv4"]=[A_conv4,None]
  histograms["W_A_conv4"]=W_A_conv4
  histograms["b_A_conv4"]=b_A_conv4

  W_A_conv5 = deep_dive.weight_variable_scaling([3,3,8,12], name='W_A_conv5')
  b_A_conv5 = deep_dive.bias_variable([12])
  A_conv5 = deep_dive.conv2d(A_conv4, W_A_conv5,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv5
  print A_conv5
  features["A_conv5"]=[A_conv5,None]
  histograms["W_A_conv5"]=W_A_conv5
  histograms["b_A_conv5"]=b_A_conv5

  W_A_conv6 = deep_dive.weight_variable_scaling([3,3,12,16], name='W_A_conv6')
  b_A_conv6 = deep_dive.bias_variable([16])
  A_conv6 = deep_dive.conv2d(A_conv5, W_A_conv6,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv6
  print A_conv6
  features["A_conv6"]=[A_conv6,None]
  histograms["W_A_conv6"]=W_A_conv6
  histograms["b_A_conv6"]=b_A_conv6

  A_concat = tf.concat(3, [A_conv1,A_conv3,A_conv6])

  W_A_conv7 = deep_dive.weight_variable_scaling([1,1,28,16], name='W_A_conv7')
  b_A_conv7 = deep_dive.bias_variable([16])
  A_conv7 = deep_dive.conv2d(A_concat, W_A_conv7,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv7
  print A_conv7
  features["A_conv7"]=[A_conv7,None]
  histograms["W_A_conv7"]=W_A_conv7
  histograms["b_A_conv7"]=b_A_conv7

  A_relu = tf.nn.relu(A_conv7+ B_relu)
  
  features["A_relu"]=[A_relu,None]



  W_C_conv1 = deep_dive.weight_variable_scaling([1,1,16,32], name='W_C_conv1')
  b_C_conv1 = deep_dive.bias_variable([32])
  C_conv1 = deep_dive.conv2d(A_relu, W_C_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv1
  print C_conv1
  features["C_conv1"]=[C_conv1,None]
  histograms["W_C_conv1"]=W_C_conv1
  histograms["b_C_conv1"]=b_C_conv1


  W_C_conv2 = deep_dive.weight_variable_scaling([1,1,16,32], name='W_C_conv2')
  b_C_conv2 = deep_dive.bias_variable([32])
  C_conv2 = deep_dive.conv2d(A_relu, W_C_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv2
  print C_conv2
  features["C_conv2"]=[C_conv2,None]
  histograms["W_C_conv2"]=W_C_conv2
  histograms["b_C_conv2"]=b_C_conv2

  W_C_conv3 = deep_dive.weight_variable_scaling([1,7,32,32], name='W_C_conv3')
  b_C_conv3 = deep_dive.bias_variable([32])
  C_conv3 = deep_dive.conv2d(C_conv2, W_C_conv3,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv3
  print C_conv3
  features["C_conv3"]=[C_conv3,None]
  histograms["W_C_conv3"]=W_C_conv3
  histograms["b_C_conv3"]=b_C_conv3

  
  W_C_conv4 = deep_dive.weight_variable_scaling([7,1,32,32], name='W_C_conv4')
  b_C_conv4 = deep_dive.bias_variable([32])
  C_conv4 = deep_dive.conv2d(C_conv3, W_C_conv4,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv4
  print C_conv4
  features["C_conv4"]=[C_conv4,None]
  histograms["W_C_conv4"]=W_C_conv4
  histograms["b_C_conv4"]=b_C_conv4

  C_concat = tf.concat(3, [C_conv1,C_conv4])

  W_C_conv5 = deep_dive.weight_variable_scaling([1,1,64,16], name='W_C_conv5')
  b_C_conv5 = deep_dive.bias_variable([16])
  C_conv5 = deep_dive.conv2d(C_concat, W_C_conv5,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv5
  print C_conv5
  features["C_conv5"]=[C_conv5,None]
  histograms["W_C_conv5"]=W_C_conv5
  histograms["b_C_conv5"]=b_C_conv5
  

  W_conv2 = deep_dive.weight_variable_scaling([3,3,16,3],name='W_conv2')
  b_conv2 = deep_dive.bias_variable([3])
  conv2 = deep_dive.conv2d(C_conv5 + A_relu, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2

  one_constant = tf.constant(1)
  C_brelu = tf.minimum(tf.to_float(one_constant), tf.nn.relu(conv2, name = "relu"), name = "brelu")

  features["C_relu"]=[C_brelu,None]
  return C_brelu,dropoutDict,features,scalars,histograms
