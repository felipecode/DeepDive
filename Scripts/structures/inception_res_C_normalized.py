
""" TODO: FINISH """

def inception_res_C(tf, x,base_name,training=True):

  from deep_dive import DeepDive
  deep_dive = DeepDive()
  features={}
  scalars={}
  histograms={}

  x_image=x
  W_C_conv1 = deep_dive.weight_variable_scaling([1,1,16,32], name='W_C_conv1'+base_name)
  C_conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_C_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv1
  features["C_conv1"+base_name]=[C_conv1,None]
  histograms["W_C_conv1"+base_name]=W_C_conv1


  W_C_conv2 = deep_dive.weight_variable_scaling([1,1,16,32], name='W_C_conv2'+base_name)
  C_conv2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_C_conv2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv2
  features["C_conv2"+base_name]=[C_conv2,None]
  histograms["W_C_conv2"+base_name]=W_C_conv2

  W_C_conv3 = deep_dive.weight_variable_scaling([1,7,32,32], name='W_C_conv3'+base_name)
  C_conv3 = tf.contrib.layers.batch_norm(deep_dive.conv2d(C_conv2, W_C_conv3,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv3
  features["C_conv3"+base_name]=[C_conv3,None]
  histograms["W_C_conv3"+base_name]=W_C_conv3

  
  W_C_conv4 = deep_dive.weight_variable_scaling([7,1,32,32], name='W_C_conv4'+base_name)
  C_conv4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(C_conv3, W_C_conv4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv4
  features["C_conv4"+base_name]=[C_conv4,None]
  histograms["W_C_conv4"+base_name]=W_C_conv4

  C_concat = tf.concat([C_conv1,C_conv4], 3)

  W_C_conv5 = deep_dive.weight_variable_scaling([1,1,64,16], name='W_C_conv5'+base_name)
  C_conv5 = tf.contrib.layers.batch_norm(deep_dive.conv2d(C_concat, W_C_conv5,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv5
  features["C_conv5"+base_name]=[C_conv5,None]
  histograms["W_C_conv5"+base_name]=W_C_conv5
  
  C_relu = tf.nn.relu(C_conv5 + x_image)
  features["C_relu"+base_name]=[C_relu,None]
  return C_relu,features,histograms
