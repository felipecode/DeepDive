"""
Returns the last tensor of the network's structure
followed dropout and features dictionaries to be summarised
Input is tensorflow class and an input placeholder.  
"""

def create_structure(tf, x, input_size,dropout,training=True):
 
  """Deep dive libs"""
  from deep_dive import DeepDive
  
  deep_dive = DeepDive()

  dropoutDict={}

  features={}
  scalars={}
  histograms={}

  x_image=x

  shapeScale1=x_image.get_shape()
  shapeScale1=shapeScale1.as_list()
  x_imageScale2=tf.image.resize_images(x_image, [shapeScale1[1]/2, shapeScale1[2]/2], method=0, align_corners=False)
  x_imageScale4=tf.image.resize_images(x_image, [shapeScale1[1]/4, shapeScale1[2]/4], method=0, align_corners=False)
  x_imageScale8=tf.image.resize_images(x_image, [shapeScale1[1]/8, shapeScale1[2]/8], method=0, align_corners=False)


  W_conv = deep_dive.weight_variable_scaling([7,7,3,16], name='W_conv')
  conv = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_conv,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv_scale2 = deep_dive.weight_variable_scaling([7,7,3,16], name='W_conv_scale2')
  conv_scale2 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(x_imageScale2, W_conv_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv_scale4 = deep_dive.weight_variable_scaling([7,7,3,16], name='W_conv_scale4')
  conv_scale4 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(x_imageScale4, W_conv_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv_scale8 = deep_dive.weight_variable_scaling([7,7,3,16], name='W_conv_scale8')
  conv_scale8 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(x_imageScale8, W_conv_scale8,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  print conv

  W_conv1 = deep_dive.weight_variable_scaling([7,1,16,16], name='W_conv1')
  conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(conv, W_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv1_scale2 = deep_dive.weight_variable_scaling([7,1,16,16], name='W_conv1_scale2')
  conv1_scale2 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(conv_scale2, W_conv1_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv1_scale4 = deep_dive.weight_variable_scaling([7,1,16,16], name='W_conv1_scale4')
  conv1_scale4 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(conv_scale4, W_conv1_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv1_scale8 = deep_dive.weight_variable_scaling([7,1,16,16], name='W_conv1_scale8')
  conv1_scale8 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(conv_scale8, W_conv1_scale8,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))


  W_conv2 = deep_dive.weight_variable_scaling([1,7,16,16], name='W_conv2')
  conv2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv1, W_conv2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  
  W_conv2_scale2 = deep_dive.weight_variable_scaling([1,7,16,16], name='W_conv2_scale2')
  conv2_scale2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv1_scale2, W_conv2_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv2_scale4 = deep_dive.weight_variable_scaling([1,7,16,16], name='W_conv2_scale4')
  conv2_scale4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv1_scale4, W_conv2_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv2_scale8 = deep_dive.weight_variable_scaling([1,7,16,16], name='W_conv2_scale8')
  conv2_scale8 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv1_scale8, W_conv2_scale8,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)



  relu=tf.nn.relu(conv+conv2)
  relu_scale2=tf.nn.relu(conv_scale2+conv2_scale2)
  relu_scale4=tf.nn.relu(conv_scale4+conv2_scale4)
  relu_scale8=tf.nn.relu(conv_scale8+conv2_scale8)

  print relu

  W_conv3 = deep_dive.weight_variable_scaling([7,1,16,16], name='W_conv3')
  conv3 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(relu, W_conv3,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv3_scale2 = deep_dive.weight_variable_scaling([7,1,16,16], name='W_conv3_scale2')
  conv3_scale2 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(relu_scale2, W_conv3_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv3_scale4 = deep_dive.weight_variable_scaling([7,1,16,16], name='W_conv3_scale4')
  conv3_scale4 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(relu_scale4, W_conv3_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv3_scale8 = deep_dive.weight_variable_scaling([7,1,16,16], name='W_conv3_scale8')
  conv3_scale8 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(relu_scale8, W_conv3_scale8,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))


  W_conv4 = deep_dive.weight_variable_scaling([1,7,16,16], name='W_conv4')
  conv4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv3, W_conv4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  
  W_conv4_scale2 = deep_dive.weight_variable_scaling([1,7,16,16], name='W_conv4_scale2')
  conv4_scale2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv3_scale2, W_conv4_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv4_scale4 = deep_dive.weight_variable_scaling([1,7,16,16], name='W_conv4_scale4')
  conv4_scale4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv3_scale4, W_conv4_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv4_scale8 = deep_dive.weight_variable_scaling([1,7,16,16], name='W_conv4_scale8')
  conv4_scale8 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv3_scale8, W_conv4_scale8,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  relu2=tf.nn.relu(relu+conv4)
  relu2_scale2=tf.nn.relu(relu_scale2+conv4_scale2)
  relu2_scale4=tf.nn.relu(relu_scale4+conv4_scale4)
  relu2_scale8=tf.nn.relu(relu_scale8+conv4_scale8)

  print relu2

  W_conv5 = deep_dive.weight_variable_scaling([5,1,16,16], name='W_conv5')
  conv5 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(relu2, W_conv5,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv5_scale2 = deep_dive.weight_variable_scaling([5,1,16,16], name='W_conv5_scale2')
  conv5_scale2 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(relu2_scale2, W_conv5_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv5_scale4 = deep_dive.weight_variable_scaling([5,1,16,16], name='W_conv5_scale4')
  conv5_scale4 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(relu2_scale4, W_conv5_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))

  W_conv5_scale8 = deep_dive.weight_variable_scaling([5,1,16,16], name='W_conv5_scale8')
  conv5_scale8 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(relu2_scale8, W_conv5_scale8,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))


  W_conv6 = deep_dive.weight_variable_scaling([1,5,16,16], name='W_conv6')
  conv6 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv5, W_conv6,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  
  W_conv6_scale2 = deep_dive.weight_variable_scaling([1,5,16,16], name='W_conv6_scale2')
  conv6_scale2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv5_scale2, W_conv6_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv6_scale4 = deep_dive.weight_variable_scaling([1,5,16,16], name='W_conv6_scale4')
  conv6_scale4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv5_scale4, W_conv6_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv6_scale8 = deep_dive.weight_variable_scaling([1,5,16,16], name='W_conv6_scale8')
  conv6_scale8 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv5_scale8, W_conv6_scale8,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  relu3=tf.nn.relu(relu2+conv6)
  relu3_scale2=tf.nn.relu(relu2_scale2+conv6_scale2)
  relu3_scale4=tf.nn.relu(relu2_scale4+conv6_scale4)
  relu3_scale8=tf.nn.relu(relu2_scale8+conv6_scale8)


  W_conv7 = deep_dive.weight_variable_scaling([3,3,16,3], name='W_conv7')
  conv7 = tf.contrib.layers.batch_norm(deep_dive.conv2d(relu3, W_conv7,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv7_scale2 = deep_dive.weight_variable_scaling([3,3,16,3], name='W_conv7_scale2')
  conv7_scale2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(relu3_scale2, W_conv7_scale2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv7_scale4 = deep_dive.weight_variable_scaling([3,3,16,3], name='W_conv7_scale4')
  conv7_scale4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(relu3_scale4, W_conv7_scale4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_conv7_scale8 = deep_dive.weight_variable_scaling([3,3,16,3], name='W_conv7_scale8')
  conv7_scale8 = tf.contrib.layers.batch_norm(deep_dive.conv2d(relu3_scale8, W_conv7_scale8,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  print relu2

  conv7_scale2=tf.image.resize_images(conv7_scale2, shapeScale1[1:3], method=0, align_corners=False)
  conv7_scale4=tf.image.resize_images(conv7_scale4, shapeScale1[1:3], method=0, align_corners=False)
  conv7_scale8=tf.image.resize_images(conv7_scale8, shapeScale1[1:3], method=0, align_corners=False)

  result = conv7_scale8 + conv7_scale4 + conv7_scale2 + conv7 +x_image

  one_constant = tf.constant(1)
  brelu = tf.minimum(tf.to_float(one_constant), tf.nn.relu(result, name = "relu"), name = "brelu")
  
  
  return brelu,dropoutDict,features,scalars,histograms
  