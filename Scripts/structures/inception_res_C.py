
""" TODO: FINISH """

def inception_res_C(tf,input_layer,input_channels, intermediate ,name_layer):

  from deep_dive import DeepDive
  deep_dive = DeepDive()
  features={}
  scalars={}
  histograms={}


  W_C_conv1 = deep_dive.weight_xavi_init([1,1,input_channels,intermediate[0]], name='W_'+name_layer +'_conv1')
  b_C_conv1 = deep_dive.bias_variable([intermediate[0]])
  C_conv1 = deep_dive.conv2d(input_layer, W_C_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv1
  print C_conv1
  #features["C_conv1"]=C_conv1
  #histograms["W_C_conv1"]=W_C_conv1
  #histograms["b_C_conv1"]=b_C_conv1


  W_C_conv2 = deep_dive.weight_xavi_init([1,1,input_channels,intermediate[1]], name='W_'+name_layer +'_conv2')
  b_C_conv2 = deep_dive.bias_variable([intermediate[1]])
  C_conv2 = deep_dive.conv2d(input_layer, W_C_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv2
  print C_conv2
  features["C_conv2"]=C_conv2
#  histograms["W_C_conv2"]=W_C_conv2
#  histograms["b_C_conv2"]=b_C_conv2

  W_C_conv3 = deep_dive.weight_xavi_init([1,7,intermediate[1],intermediate[2]], name='W_'+name_layer +'_conv3')
  b_C_conv3 = deep_dive.bias_variable([intermediate[2]])
  C_conv3 = deep_dive.conv2d(C_conv2, W_C_conv3,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv3
  print C_conv3
  features["C_conv3"]=C_conv3
#  histograms["W_C_conv3"]=W_C_conv3
#  histograms["b_C_conv3"]=b_C_conv3

  
  W_C_conv4 = deep_dive.weight_xavi_init([7,1,intermediate[2],intermediate[3]], name='W_'+name_layer +'_conv4')
  b_C_conv4 = deep_dive.bias_variable([intermediate[3]])
  C_conv4 = deep_dive.conv2d(C_conv3, W_C_conv4,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv4
  print C_conv4
 # features["C_conv4"]=C_conv4
#  histograms["W_C_conv4"]=W_C_conv4
#  histograms["b_C_conv4"]=b_C_conv4

  C_concat = tf.concat(3, [C_conv1,C_conv4])

  W_C_conv5 = deep_dive.weight_xavi_init([1,1,intermediate[0]+intermediate[3],input_channels], name='W_'+name_layer +'_conv5')
  b_C_conv5 = deep_dive.bias_variable([input_channels])
  C_conv5 = deep_dive.conv2d(C_concat, W_C_conv5,strides=[1, 1, 1, 1], padding='SAME') + b_C_conv5
  print C_conv5
#  features["C_conv5"]=C_conv5
#  histograms["W_C_conv5"]=W_C_conv5
 # histograms["b_C_conv5"]=b_C_conv5
  return C_conv5
