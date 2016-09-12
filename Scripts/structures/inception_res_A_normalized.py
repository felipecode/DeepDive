
def inception_res_A(tf, x,base_name,training=True):


  from deep_dive import DeepDive
  deep_dive = DeepDive()
  features={}
  scalars={}
  histograms={}

  x_image=x
  W_A_conv1 = deep_dive.weight_variable_scaling([1,1,16,6], name='W_A_conv1'+base_name)
  A_conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_A_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv1
  features["A_conv1"+base_name]=[A_conv1,None]
  histograms["W_A_conv1"+base_name]=W_A_conv1

  W_A_conv2 = deep_dive.weight_variable_scaling([1,1,16,6], name='W_A_conv2'+base_name)
  A_conv2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_A_conv2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv2
  features["A_conv2"+base_name]=[A_conv2,None]
  histograms["W_A_conv2"+base_name]=W_A_conv2

  W_A_conv3 = deep_dive.weight_variable_scaling([3,3,6,6], name='W_A_conv3'+base_name)
  A_conv3 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_conv2, W_A_conv3,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv3
  features["A_conv3"+base_name]=[A_conv3,None]
  histograms["W_A_conv3"+base_name]=W_A_conv3

  W_A_conv4 = deep_dive.weight_variable_scaling([1,1,16,8], name='W_A_conv4'+base_name)
  A_conv4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_A_conv4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  features["A_conv4"+base_name]=[A_conv4,None]
  histograms["W_A_conv4"+base_name]=W_A_conv4

  W_A_conv5 = deep_dive.weight_variable_scaling([3,3,8,12], name='W_A_conv5'+base_name)
  A_conv5 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_conv4, W_A_conv5,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv5
  features["A_conv5"+base_name]=[A_conv5,None]
  histograms["W_A_conv5"+base_name]=W_A_conv5

  W_A_conv6 = deep_dive.weight_variable_scaling([3,3,12,16], name='W_A_conv6'+base_name)
  A_conv6 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_conv5, W_A_conv6,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv6
  features["A_conv6"+base_name]=[A_conv6,None]
  histograms["W_A_conv6"+base_name]=W_A_conv6

  A_concat = tf.concat(3, [A_conv1,A_conv3,A_conv6])

  W_A_conv7 = deep_dive.weight_variable_scaling([1,1,28,16], name='W_A_conv7'+base_name)
  A_conv7 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_concat, W_A_conv7,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv7
  features["A_conv7"+base_name]=[A_conv7,None]
  histograms["W_A_conv7"+base_name]=W_A_conv7

  A_relu = tf.nn.relu(A_conv7+ x_image)
  
  features["A_relu"+base_name]=[A_relu,None]
  return A_relu,features,histograms