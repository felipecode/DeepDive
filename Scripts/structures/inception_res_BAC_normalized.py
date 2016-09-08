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

  
  """ Scale 1 """

  features={}
  scalars={}
  histograms={}
  # tf.contrib.layers.batch_norm(inputs,decay=0.999,center=True,scale=False,epsilon=0.001,
  #                 activation_fn=None,updates_collections=ops.GraphKeys.UPDATE_OPS,
  #                is_training=True,reuse=None,variables_collections=None, outputs_collections=None, trainable=True, scope=None)
  """
  Args:
    inputs: a tensor of size `[batch_size, height, width, channels]`
            or `[batch_size, channels]`.
    decay: decay for the moving average.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: Optional activation function.
    updates_collections: collections to collect the update ops for computation.
      If None, a control dependency would be added to make sure the updates are
      computed.
    is_training: whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_op_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """

  x_image = x#tf.contrib.layers.batch_norm(x,center=True,updates_collections=None,scale=True,is_training=training)
  W_conv1 = deep_dive.weight_variable_scaling([3,3,3,16],name='W_conv1')
  conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)

  W_B_conv1 = deep_dive.weight_variable_scaling([1,1,16,24], name='W_B_conv1')
  B_conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv1, W_B_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv1
  features["B_conv1"] = [B_conv1,None]
  histograms["W_B_conv1"]=W_B_conv1

  W_B_conv2 = deep_dive.weight_variable_scaling([1,1,16,24], name='W_B_conv2')
  B_conv2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(conv1, W_B_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv2
  features["B_conv2"] = [B_conv2,None]
  histograms["W_B_conv2"]=W_B_conv2

  W_B_conv3 = deep_dive.weight_variable_scaling([1,3,24,28], name='W_B_conv3')
  B_conv3 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_conv2, W_B_conv3,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv3
  features["B_conv3"] = [B_conv3,None]
  histograms["W_B_conv3"]=W_B_conv3

  W_B_conv4 = deep_dive.weight_variable_scaling([3,1,28,32], name='W_B_conv4')
  B_conv4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_conv3, W_B_conv4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv4
  features["B_conv4"] = [B_conv4,None]
  histograms["W_B_conv4"]=W_B_conv4

  B_concat = tf.concat(3, [B_conv1,B_conv4])

  W_B_conv5 = deep_dive.weight_variable_scaling([1,1,56,16], name='W_B_conv5')
  B_conv5 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_concat, W_B_conv5,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print B_conv5
  features["B_conv5"]=[B_conv5,None]
  histograms["W_B_conv5"]=W_B_conv5

  B_relu = tf.nn.relu(B_conv5+ conv1)
  features["B_relu"]=[B_relu,None]

  W_A_conv1 = deep_dive.weight_variable_scaling([1,1,16,6], name='W_A_conv1')
  A_conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_relu, W_A_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv1
  features["A_conv1"]=[A_conv1,None]
  histograms["W_A_conv1"]=W_A_conv1

  W_A_conv2 = deep_dive.weight_variable_scaling([1,1,16,6], name='W_A_conv2')
  A_conv2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_relu, W_A_conv2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv2
  features["A_conv2"]=[A_conv2,None]
  histograms["W_A_conv2"]=W_A_conv2

  W_A_conv3 = deep_dive.weight_variable_scaling([3,3,6,6], name='W_A_conv3')
  A_conv3 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_conv2, W_A_conv3,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv3
  features["A_conv3"]=[A_conv3,None]
  histograms["W_A_conv3"]=W_A_conv3

  W_A_conv4 = deep_dive.weight_variable_scaling([1,1,16,8], name='W_A_conv4')
  A_conv4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(B_relu, W_A_conv4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  features["A_conv4"]=[A_conv4,None]
  histograms["W_A_conv4"]=W_A_conv4

  W_A_conv5 = deep_dive.weight_variable_scaling([3,3,8,12], name='W_A_conv5')
  A_conv5 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_conv4, W_A_conv5,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv5
  features["A_conv5"]=[A_conv5,None]
  histograms["W_A_conv5"]=W_A_conv5

  W_A_conv6 = deep_dive.weight_variable_scaling([3,3,12,16], name='W_A_conv6')
  A_conv6 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_conv5, W_A_conv6,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv6
  features["A_conv6"]=[A_conv6,None]
  histograms["W_A_conv6"]=W_A_conv6

  A_concat = tf.concat(3, [A_conv1,A_conv3,A_conv6])

  W_A_conv7 = deep_dive.weight_variable_scaling([1,1,28,16], name='W_A_conv7')
  A_conv7 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_concat, W_A_conv7,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print A_conv7
  features["A_conv7"]=[A_conv7,None]
  histograms["W_A_conv7"]=W_A_conv7

  A_relu = tf.nn.relu(A_conv7+ B_relu)
  
  features["A_relu"]=[A_relu,None]



  W_C_conv1 = deep_dive.weight_variable_scaling([1,1,16,32], name='W_C_conv1')
  C_conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_relu, W_C_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv1
  features["C_conv1"]=[C_conv1,None]
  histograms["W_C_conv1"]=W_C_conv1


  W_C_conv2 = deep_dive.weight_variable_scaling([1,1,16,32], name='W_C_conv2')
  C_conv2 = tf.contrib.layers.batch_norm(deep_dive.conv2d(A_relu, W_C_conv2,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv2
  features["C_conv2"]=[C_conv2,None]
  histograms["W_C_conv2"]=W_C_conv2

  W_C_conv3 = deep_dive.weight_variable_scaling([1,7,32,32], name='W_C_conv3')
  C_conv3 = tf.contrib.layers.batch_norm(deep_dive.conv2d(C_conv2, W_C_conv3,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv3
  features["C_conv3"]=[C_conv3,None]
  histograms["W_C_conv3"]=W_C_conv3

  
  W_C_conv4 = deep_dive.weight_variable_scaling([7,1,32,32], name='W_C_conv4')
  C_conv4 = tf.contrib.layers.batch_norm(deep_dive.conv2d(C_conv3, W_C_conv4,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv4
  features["C_conv4"]=[C_conv4,None]
  histograms["W_C_conv4"]=W_C_conv4

  C_concat = tf.concat(3, [C_conv1,C_conv4])

  W_C_conv5 = deep_dive.weight_variable_scaling([1,1,64,16], name='W_C_conv5')
  C_conv5 = tf.contrib.layers.batch_norm(deep_dive.conv2d(C_concat, W_C_conv5,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  print C_conv5
  features["C_conv5"]=[C_conv5,None]
  histograms["W_C_conv5"]=W_C_conv5
  
  C_relu = tf.nn.relu(C_conv5 + A_relu)
  features["C_relu"]=[C_relu,None]
  W_conv2 = deep_dive.weight_variable_scaling([3,3,16,3],name='W_conv2')
  b_conv2 = deep_dive.bias_variable([3])

  conv2 = deep_dive.conv2d(C_relu, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2

  one_constant = tf.constant(1)

  brelu = tf.minimum(tf.to_float(one_constant), tf.nn.relu(conv2, name = "relu"), name = "brelu")
  
  
  return brelu,dropoutDict,features,scalars,histograms