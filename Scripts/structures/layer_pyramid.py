
def layer_pyramid(tf, x,base_name,training=True):


  from deep_dive import DeepDive
  deep_dive = DeepDive()
  features={}
  scalars={}
  histograms={}

  x_image=x
  shapeScale1=x_image.get_shape()
  x_imageScale2=tf.image.resize_images(x, shapeScale1[1]/2, shapeScale1[2]/2, method=0, align_corners=False)
  #shapeScale2=x_image2.get_shape()
  x_imageScale4=tf.image.resize_images(x_imageScale2, shapeScale1[1]/4, shapeScale1[2]/4, method=0, align_corners=False)
  #shapeScale4=x_imageScale4.get_shape()

  W_conv = deep_dive.weight_variable_scaling([3,3,64,64], name='W_conv1'+base_name)
  conv = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_conv,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv_scale2 = deep_dive.weight_variable_scaling([3,3,64,64], name='W_conv1_scale2'+base_name)
  conv_scale2 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(x_imageScale2, W_conv_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv_scale4 = deep_dive.weight_variable_scaling([3,3,64,64], name='W_conv1_scale4'+base_name)
  conv_scale4 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(x_imageScale4, W_conv_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  print conv
  features["conv"+base_name]=[conv,None]
  histograms["conv"+base_name]=W_conv

  W_conv2 = deep_dive.weight_variable_scaling([3,3,64,64], name='W_conv2'+base_name)
  conv2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv, W_conv2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  
  W_conv2_scale2 = deep_dive.weight_variable_scaling([3,3,64,64], name='W_conv2_scale2'+base_name)
  conv2_scale2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv_scale2, W_conv2_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv2_scale4 = deep_dive.weight_variable_scaling([3,3,64,64], name='W_conv2_scale4'+base_name)
  conv2_scale4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv_scale4, W_conv2_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)


  print conv2
  features["conv2"+base_name]=[conv2,None]
  histograms["conv2"+base_name]=W_conv2

  conv2_scale2=tf.image.resize_images(conv2_scale2, shapeScale1[1], shapeScale1[2], method=0, align_corners=False)
  conv2_scale4=tf.image.resize_images(conv2_scale4, shapeScale1[1], shapeScale1[2], method=0, align_corners=False)


  relu=tf.nn.relu(x_image+conv2+conv2_scale2+conv2_scale4)
  #relu=tf.nn.relu(x_image+(conv2_scale4+(conv2_scale2-conv2_scale4)+(conv2-conv2_scale2))

  features["conv2"+base_name]=[relu,None]
  print relu
  

  return relu,features,histograms