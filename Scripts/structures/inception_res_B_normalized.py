
def inception_res_B(tf, x,base_name,training=True):

  from deep_dive import DeepDive
  deep_dive = DeepDive()
  features={}
  scalars={}
  histograms={}

  x_image=x
  x_image = tf.contrib.layers.batch_norm(x_image,center=True,updates_collections=None,scale=True,is_training=training)
  W_B_conv1 = deep_dive.weight_variable_scaling([1,1,16,24], name='W_B_conv1'+base_name)
  B_conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_B_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv1
  features["B_conv1"+base_name] = [B_conv1,None]
  histograms["W_B_conv1"+base_name]=W_B_conv1

  W_B_conv2 = deep_dive.weight_variable_scaling([1,1,16,24], name='W_B_conv2'+base_name)
  B_conv2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_B_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv2
  features["B_conv2"+base_name] = [B_conv2,None]
  histograms["W_B_conv2"+base_name]=W_B_conv2

  W_B_conv3 = deep_dive.weight_variable_scaling([1,3,24,28], name='W_B_conv3'+base_name)
  B_conv3 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_conv2, W_B_conv3,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv3
  features["B_conv3"+base_name] = [B_conv3,None]
  histograms["W_B_conv3"+base_name]=W_B_conv3

  W_B_conv4 = deep_dive.weight_variable_scaling([3,1,28,32], name='W_B_conv4'+base_name)
  B_conv4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_conv3, W_B_conv4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv4
  features["B_conv4"+base_name] = [B_conv4,None]
  histograms["W_B_conv4"+base_name]=W_B_conv4

  B_concat = tf.concat([B_conv1,B_conv4], 3)

  W_B_conv5 = deep_dive.weight_variable_scaling([1,1,56,16], name='W_B_conv5'+base_name)
  B_conv5 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_concat, W_B_conv5,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv5
  features["B_conv5"+base_name]=[B_conv5,None]
  histograms["W_B_conv5"+base_name]=W_B_conv5

  B_relu = tf.nn.relu(B_conv5+ x_image)
  features["B_relu"+base_name]=[B_relu,None]
  return B_relu,features,histograms

