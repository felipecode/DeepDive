
def create_structure(tf, x, input_size, dropout, training=True, epsilon=1e-6):
    
    from deep_dive import DeepDive
    
    deep_dive   = DeepDive()
    features    = {}
    dropoutDict = {}
    scalars     = {}
    histograms  = {}
    x_image     = x
    base_name   = "esc_net"
    n_params = {'center':True,
        'updates_collections': None, 'scale': True, 'is_training': training}
    
    batch_size = x_image.shape[0].value
    # CONV 1
    #   INPUT: (224x224x5)
    #   3x1
    #   SAME
    #   32x
    #   OUTPUT: (224x224x32)

    #W_conv1 = deep_dive.weight_variable_scaling ( [3, 1, 3, 16], name='W_conv1'+base_name )
    #bias = deep_dive.bias_variable([16])
    conv1 = tf.contrib.layers.conv2d(x_image, 32, [3, 1], stride=(1, 1) ,padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv1
    features ["A_conv1"+base_name] = conv1
    #histograms ["W_conv1"+base_name] = W_conv1

    # CONV 2
    #   INPUT: (224x224x32)
    #   1x3
    #   SAME
    #   32x
    #   OUTPUT: (224x224x32)

    #W_conv2 = deep_dive.weight_variable_scaling ( [1, 3, 16, 16], name='W_conv2'+base_name )
    #bias = deep_dive.bias_variable([16])
    conv2 = tf.contrib.layers.conv2d(conv1, 32, [1, 3], stride=(1, 1), padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv2
    features ["A_conv2"+base_name] = conv2
    #histograms ["W_conv2"+base_name] = W_conv2

    # CONV 3
    # INPUT: (224x224x34)
    # 3x1
    # SAME
    # 32x
    # OUTPUT: (224x224x32)

    #W_conv3 = deep_dive.weight_variable_scaling ( [3, 1, 16, 16], name='W_conv3'+base_name )
    #bias = deep_dive.bias_variable([16])
    conv3 = tf.contrib.layers.conv2d(conv2, 32, [3, 1], stride=(1, 1), padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv3
    features ["A_conv3"+base_name] = conv3
    #histograms ["W_conv3"+base_name] = W_conv3

    # CONV 4
    # INPUT: (224x224x32)
    # 1x3
    # SAME
    # 32x
    # OUTPUT: (224x224x32)

    #W_conv4 = deep_dive.weight_variable_scaling ( [1, 3, 16, 16], name='W_conv4'+base_name )
    #bias = deep_dive.bias_variable([16])
    conv4 = tf.contrib.layers.conv2d(conv3, 32, [1, 3], stride=(1, 1), padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv4
    features ["A_conv4"+base_name] = conv4
    #histograms ["W_conv4"+base_name] = W_conv4

    # CONV 5
    # INPUT: (224x224x32)
    # 3x3
    # SAME
    # 64x
    # OUTPUT: (224x224x64)

    #W_conv5 = deep_dive.weight_variable_scaling ( [3, 3, 16, 32], name='W_conv5'+base_name )
    #bias = deep_dive.bias_variable([32])
    conv5 = tf.contrib.layers.conv2d(conv4, 64, [3, 3], stride=(1, 1), padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv5
    features ["A_conv5"+base_name] = conv5
    #histograms ["W_conv5"+base_name] = W_conv5

    # CONV 6
    # INPUT: (224x224x64)
    # 3x3   
    # SAME
    # 64x
    # OUTPUT: (224x224x64)
    
    #W_conv6 = deep_dive.weight_variable_scaling ( [3, 3, 32, 32], name='W_conv6'+base_name )
    #bias = deep_dive.bias_variable([32])
    conv6 = tf.contrib.layers.conv2d(conv5, 64, [3, 3], stride=(1, 1), padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv6
    features ["A_conv6"+base_name] = conv6
    #histograms ["W_conv6"+base_name] = W_conv6

    #
    # CONV 7
    # INPUT: (224x224x64)
    # 3x1
    # SAME
    # 64x
    # OUTPUT: (224x224x64)

    #W_conv7 = deep_dive.weight_variable_scaling ( [3, 3, 32, 64], name='W_conv7'+base_name )
    #bias = deep_dive.bias_variable([64])
    conv7 = tf.contrib.layers.conv2d(conv6, 64, [3, 1], stride=(1, 1), padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv7
    features ["A_conv7"+base_name] = conv7
    #histograms ["W_conv7"+base_name] = W_conv7

    # CONV 8
    # INPUT: (224x224x64)
    # 1x3
    # SAME
    # 64x
    # OUTPUT: (224x224x64)
    #W_conv8 = deep_dive.weight_variable_scaling ( [1, 1, 64, 16], name='W_conv8'+base_name )
    #bias = deep_dive.bias_variable([16])
    conv8 = tf.contrib.layers.conv2d(conv7, 64, [1, 3], stride=(1, 1), padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv8
    features ["A_conv8"+base_name] = conv8
    #histograms ["W_conv8"+base_name] = W_conv8
    
    # CONV 9
    # INPUT: (224x224x64)
    # 1x1
    # SAME
    # 1x
    # OUTPUT: (224x224x1)
    #W_conv9 = deep_dive.weight_variable_scaling ( [3, 3, 16, 16], name='W_conv9'+base_name )
    #bias = deep_dive.bias_variable([16])
    conv9 = tf.contrib.layers.conv2d(conv8, 1, [1, 1], stride=(1, 1), padding='SAME',
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    
    print conv9
    features ["A_conv9"+base_name] = conv9
    """
    # Z-POOLING 1
    # INPUT: (224x224x32)
    # 1x1x32
    # SAME
    # STRIDE 1
    # OUTPUT (224x224x1)
    pool = tf.reduce_max(conv8, reduction_indices=[3], keep_dims=True, name="Z-POOL")
    print pool
    features["pool" + base_name] = pool
    
    # FC 1
    # INPUT: (224^2)
    # OUTPUT: (2x1)
    #conv9_flat = tf.reshape(conv9, (batch_size, conv9.shape[1].value * conv9.shape[2].value))
    pool_flat = tf.reshape(pool, (batch_size, pool.shape[1].value * pool.shape[2].value))
    fc = tf.contrib.layers.fully_connected(pool_flat, 2,
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=n_params, trainable=training)
    fc = tf.reshape(fc, (batch_size, 2, 1, 1))
    print fc
    """
    maxes = tf.reshape(tf.reduce_max(conv9, (1, 2)), (batch_size, 1, 1, 1))
    normal_final = (conv9) / (maxes + epsilon)
    return normal_final, dropoutDict, features, scalars, histograms
"""
    #ARGMAX
    argmax_flat = tf.argmax(tf.reshape(conv9, [conv9.shape[0].value, -1]), axis=1)
    print(argmax_flat)
    part1 = argmax_flat // conv9.shape[2].value
    part2 = tf.mod(argmax_flat, conv9.shape[2].value)
    argmax = tf.stack([part1, part2])
    argmax = tf.transpose(argmax)
    argmax = (tf.reshape(argmax, (argmax.shape[0].value, argmax.shape[1].value, 1,1)))

    print argmax
    return argmax, dropoutDict, features, scalars, histograms

    #raise Exception()
"""

"""
    #histograms ["W_conv9"+base_name] = W_conv9
    # CONV 10
    # INPUT: (4x4x16)
    # 3x3
    # SAME
    # 1x
    # STRIDE 1x1
    # OUTPUT: (4x4x1)
    W_conv10 = deep_dive.weight_variable_scaling ( [3, 3, 16, 1], name='W_conv10'+base_name )
    #bias = deep_dive.bias_variable([1])
    conv10 = tf.contrib.layers.batch_norm(tf.nn.relu(
        tf.nn.conv2d ( conv9, W_conv10, stride=[1, 1, 1, 1] ,padding='SAME' ) ), center=True,
        updates_collections=None, scale=True, is_training=training )
    
    print conv10
    features ["A_conv10"+base_name] = conv10
    histograms ["W_conv10"+base_name] = W_conv10
    
    # AVGPOOL 1
    # INPUT: (4x2x1)
    # 2x2
    # VALID
    # STRIDE: 2x2
    # OUTPUT: (2x1x1)

    pool3 = tf.nn.avg_pool(conv10, ksize = [1, 2, 2, 1], stride = [1, 2, 2, 1], padding = 'VALID', name = 'pool3')
    print pool3
    features["pool3"] = pool3

    final = pool3
    tf_size = tf.constant((input_size[0], input_size[1]), dtype=tf.float32)
    #tf_size_x = tf.constant(input_size[0], dtype=tf.float32)
    #tf_size_y = tf.constant(input_size[1], dtype=tf.float32)
    #print(input_size)
    brelu = tf.expand_dims(tf.expand_dims(tf.minimum(final[:,:,0,0], tf_size), 2), 3)
    print brelu
"""

    
    #return brelu, dropoutDict, features, scalars, histograms