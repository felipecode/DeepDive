"""
Returns the last tensor of the network's structure
followed dropout and features dictionaries to be summarised
Input is tensorflow class and an input placeholder.  
"""
from residual_elu import *

def create_structure(tf, x, input_size,dropout,training=True):
 
  """Deep dive libs"""
  from deep_dive import DeepDive
  
  deep_dive = DeepDive()



  dropoutDict={}

  
  """ Scale 1 """

  features={}
  scalars={}
  histograms={}


  x_image=x
  #x_image = tf.contrib.layers.batch_norm(x_image,center=True,updates_collections=None,scale=True,is_training=training)
  W_conv1 = deep_dive.weight_variable_scaling([7,7,3,64],name='W_conv1')
  b_conv1 = deep_dive.bias_variable([64])
  #conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training))
  conv1 = tf.nn.elu(deep_dive.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME')+b_conv1)

  last_layer1,features1,histograms1=residual(tf=tf, x=conv1,training=training,base_name="layer1")
  features.update(features1)
  histograms.update(histograms1)

  last_layer2,features2,histograms2=residual(tf=tf, x=last_layer1,training=training,base_name="layer2")
  features.update(features2)
  histograms.update(histograms2)

  last_layer3,features3,histograms3=residual(tf=tf, x=last_layer2,training=training,base_name="layer3")
  features.update(features3)
  histograms.update(histograms3)

  last_layer4,features4,histograms4=residual(tf=tf, x=last_layer3,training=training,base_name="layer4")
  features.update(features4)
  histograms.update(histograms4)

  last_layer5,features5,histograms5=residual(tf=tf, x=last_layer4,training=training,base_name="layer5")
  features.update(features5)
  histograms.update(histograms5)

  last_layer6,features6,histograms6=residual(tf=tf, x=last_layer5,training=training,base_name="layer6")
  features.update(features6)
  histograms.update(histograms6)

  last_layer7,features7,histograms7=residual(tf=tf, x=last_layer6,training=training,base_name="layer7")
  features.update(features7)
  histograms.update(histograms7)

  last_layer8,features8,histograms8=residual(tf=tf, x=last_layer7,training=training,base_name="layer8")
  features.update(features8)
  histograms.update(histograms8)

  last_layer9,features9,histograms9=residual(tf=tf, x=last_layer8,training=training,base_name="layer9")
  features.update(features9)
  histograms.update(histograms9)

  last_layer10,features10,histograms10=residual(tf=tf, x=last_layer9,training=training,base_name="layer10")
  features.update(features10)
  histograms.update(histograms10)

  last_layer11,features11,histograms11=residual(tf=tf, x=last_layer10,training=training,base_name="layer11")
  features.update(features11)
  histograms.update(histograms11)

  last_layer12,features12,histograms12=residual(tf=tf, x=last_layer11,training=training,base_name="layer12")
  features.update(features12)
  histograms.update(histograms12)


  W_conv2 = deep_dive.weight_variable_scaling([7,7,64,3],name='W_conv2')
  b_conv2 = deep_dive.bias_variable([3])

  conv2 = deep_dive.conv2d(last_layer12+conv1, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2

  one_constant = tf.constant(1)

  brelu = tf.minimum(tf.to_float(one_constant), tf.nn.elu(conv2, name = "relu"), name = "brelu")
  
  
  return brelu,dropoutDict,features,scalars,histograms
