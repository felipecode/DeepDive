from deep_dive import DeepDive

def discriminator_layer(tf, x, base_name, n, depth, stride, training=True):
  deep_dive = DeepDive()
  features={}
  scalars={}
  histograms={}

  x_image = x

  W_conv = deep_dive.weight_variable_scaling([3, 3, depth, n], name = 'W_conv' + base_name)
  b_conv = deep_dive.bias_variable([n])

  histograms["W_conv" + base_name] = W_conv
  histograms["b_conv" + base_name] = b_conv

  conv = deep_dive.conv2d(x_image, W_conv, strides = [1, stride, stride, 1], padding = 'VALID') + b_conv
  features["conv" + base_name] = [conv, None]

  leaky_relu = tf.maximum(0.1*conv, conv)

  normalized = tf.contrib.layers.batch_norm(leaky_relu, center = True, updates_collections = None, scale = True, is_training = training)

  return normalized, features, histograms
