
def inception_res_A(tf,input_layer,input_channels,intermediate,name_layer):


  from deep_dive import DeepDive
  deep_dive = DeepDive()
  features={}
  scalars={}
  histograms={}

  W_A_conv1 = deep_dive.weight_xavi_init([1,1,input_channels,intermediate[0]], name='W_'+name_layer +'_conv1')
  b_A_conv1 = deep_dive.bias_variable([intermediate[0]])
  A_conv1 = deep_dive.conv2d(input_layer, W_A_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv1
  print A_conv1
  features[name_layer +"_conv1"]=A_conv1
  histograms["W_"+name_layer +"_conv1"]=W_A_conv1
  histograms["b_"+name_layer +"_conv1"]=b_A_conv1

  W_A_conv2 = deep_dive.weight_xavi_init([1,1,input_channels,intermediate[1]], name='W_' + name_layer +'_conv2')
  b_A_conv2 = deep_dive.bias_variable([intermediate[1]])
  A_conv2 = deep_dive.conv2d(input_layer, W_A_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv2
  print A_conv2
  features[name_layer +"_conv2"]=A_conv2
  histograms["W_"+name_layer +"_conv2"]=W_A_conv2
  histograms["b_"+name_layer +"_conv2"]=b_A_conv2

  W_A_conv3 = deep_dive.weight_xavi_init([3,3,intermediate[1],intermediate[2]], name='W_'+name_layer +'_conv3')
  b_A_conv3 = deep_dive.bias_variable([intermediate[2]])
  A_conv3 = deep_dive.conv2d(A_conv2, W_A_conv3,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv3
  print A_conv3
  features[name_layer +"_conv3"]=A_conv3
  histograms["W_"+name_layer +"_conv3"]=W_A_conv3
  histograms["b_"+name_layer +"_conv3"]=b_A_conv3

  W_A_conv4 = deep_dive.weight_xavi_init([1,1,input_channels,intermediate[3]], name='W_'+name_layer +'_conv4')
  b_A_conv4 = deep_dive.bias_variable([intermediate[3]])
  A_conv4 = deep_dive.conv2d(input_layer, W_A_conv4,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv4
  print A_conv4
  features[name_layer +"_conv4"]=A_conv4
  histograms["W_"+name_layer +"_conv4"]=W_A_conv4
  histograms["b_"+name_layer +"_conv4"]=b_A_conv4

  W_A_conv5 = deep_dive.weight_xavi_init([3,3,intermediate[3],intermediate[4]], name='W_'+name_layer +'_conv5')
  b_A_conv5 = deep_dive.bias_variable([intermediate[4]])
  A_conv5 = deep_dive.conv2d(A_conv4, W_A_conv5,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv5
  print A_conv5
  features[name_layer +"_conv5"]=A_conv5
  histograms["W_"+name_layer +"_conv5"]=W_A_conv5
  histograms["b_"+name_layer +"_conv5"]=b_A_conv5

  W_A_conv6 = deep_dive.weight_xavi_init([3,3,intermediate[4],intermediate[5]], name='W_'+name_layer +'_conv6')
  b_A_conv6 = deep_dive.bias_variable([intermediate[5]])
  A_conv6 = deep_dive.conv2d(A_conv5, W_A_conv6,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv6
  print A_conv6
  features[name_layer +"_conv6"]=A_conv6
  histograms["W_"+name_layer +"_conv6"]=W_A_conv6
  histograms["b_"+name_layer +"_conv6"]=b_A_conv6

  A_concat = tf.concat(3, [A_conv1,A_conv3,A_conv6])

  W_A_conv7 = deep_dive.weight_xavi_init([1,1,intermediate[0]+intermediate[2] +intermediate[5],input_channels], name='W_'+name_layer +'_conv7')
  b_A_conv7 = deep_dive.bias_variable([input_channels])
  A_conv7 = deep_dive.conv2d(A_concat, W_A_conv7,strides=[1, 1, 1, 1], padding='SAME') + b_A_conv7
  print A_conv7
  return A_conv7,features,histograms