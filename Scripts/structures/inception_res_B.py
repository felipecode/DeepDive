
def inception_res_B(tf,input_layer,input_channels,intermediate,name_layer):

  from deep_dive import DeepDive
  deep_dive = DeepDive()
  features={}
  scalars={}
  histograms={}

  W_B_conv1 = deep_dive.weight_xavi_init([1,1,input_channels,intermediate[0]], name='W_'+name_layer+'_conv1')
  b_B_conv1 = deep_dive.bias_variable([intermediate[0]])
  B_conv1 = deep_dive.conv2d(input_layer, W_B_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv1
  print B_conv1
  features[name_layer +"_conv1"]=B_conv1
  histograms["W_"+name_layer +"_conv1"]=W_B_conv1
  histograms["b_"+name_layer +"_conv1"]=b_B_conv1


  W_B_conv2 = deep_dive.weight_xavi_init([1,1,input_channels,intermediate[1]], name='W_'+name_layer+'_conv2')
  b_B_conv2 = deep_dive.bias_variable([intermediate[1]])
  B_conv2 = deep_dive.conv2d(input_layer, W_B_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv2
  print B_conv2
  features[name_layer +"_conv2"]=B_conv2
  histograms["W_"+name_layer +"_conv2"]=W_B_conv2
  histograms["b_"+name_layer +"_conv2"]=b_B_conv2

  W_B_conv3 = deep_dive.weight_xavi_init([1,input_channels,intermediate[1],intermediate[2]], name='W_'+name_layer +'_conv3')
  b_B_conv3 = deep_dive.bias_variable([intermediate[2]])
  B_conv3 = deep_dive.conv2d(B_conv2, W_B_conv3,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv3
  print B_conv3
  features[name_layer +"_conv3"]=B_conv3
  histograms["W_"+name_layer +"_conv3"]=W_B_conv3
  histograms["b_"+name_layer +"_conv3"]=b_B_conv3

  
  W_B_conv4 = deep_dive.weight_xavi_init([3,1,intermediate[2],intermediate[3]], name='W_'+name_layer + '_conv4')
  b_B_conv4 = deep_dive.bias_variable([intermediate[3]])
  B_conv4 = deep_dive.conv2d(B_conv3, W_B_conv4,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv4
  print B_conv4
  features[name_layer +"_conv4"]=B_conv4
  histograms["W_"+name_layer +"_conv4"]=W_B_conv4
  histograms["b_"+name_layer +"_conv4"]=b_B_conv4

  B_concat = tf.concat(3, [B_conv1,B_conv4])

  W_B_conv5 = deep_dive.weight_xavi_init([1,1,intermediate[0] + intermediate[3],input_channels], name='W_'+name_layer +'_conv5')
  b_B_conv5 = deep_dive.bias_variable([input_channels])
  B_conv5 = deep_dive.conv2d(B_concat, W_B_conv5,strides=[1, 1, 1, 1], padding='SAME') + b_B_conv5
  print B_conv5
  features[name_layer +"_conv5"]=B_conv5
  histograms["W_"+name_layer +"_conv5"]=W_B_conv5
  histograms["b_"+name_layer +"_conv5"]=b_B_conv5
  return B_conv5,features,histograms

