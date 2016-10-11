"""
Returns the last tensor of the network's structure
followed dropout and features dictionaries to be summarised
Input is tensorflow class and an input placeholder.  
"""
from inception_res_A_normalized import *

from inception_res_B_normalized import *

from inception_res_C_normalized import *
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
  x_image = tf.contrib.layers.batch_norm(x_image,center=True,updates_collections=None,scale=True,is_training=training)
  W_conv1 = deep_dive.weight_variable_scaling([3,3,3,16],name='W_conv1')
  conv1 = tf.contrib.layers.batch_norm(deep_dive.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME'),center=True,updates_collections=None,scale=True,is_training=training)
  last_layerB,featuresB,histogramsB=inception_res_B(tf=tf, x=conv1,training=training,base_name="first")
  features.update(featuresB)
  histograms.update(histogramsB)

  last_layerA,featuresA,histogramsA=inception_res_A(tf=tf, x=last_layerB,training=training,base_name="first")
  features.update(featuresA)
  histograms.update(histogramsA)

  last_layerC,featuresC,histogramsC=inception_res_C(tf=tf, x=last_layerA,training=training,base_name="first")
  features.update(featuresC)
  histograms.update(histogramsC)

  last_layerB2,featuresB2,histogramsB2=inception_res_B(tf=tf, x=last_layerC,training=training,base_name="second")
  features.update(featuresB2)
  histograms.update(histogramsB2)

  last_layerA2,featuresA2,histogramsA2=inception_res_A(tf=tf, x=last_layerB2,training=training,base_name="second")
  features.update(featuresA2)
  histograms.update(histogramsA2)

  last_layerC2,featuresC2,histogramsC2=inception_res_C(tf=tf, x=last_layerA2,training=training,base_name="second")
  features.update(featuresC2)
  histograms.update(histogramsC2)


  W_conv2 = deep_dive.weight_variable_scaling([3,3,16,3],name='W_conv2')
  b_conv2 = deep_dive.bias_variable([3])

  conv2 = deep_dive.conv2d(last_layerC2, W_conv2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2

  one_constant = tf.constant(1)

  brelu = tf.minimum(tf.to_float(one_constant), tf.nn.relu(conv2, name = "relu"), name = "brelu")
  
  
  return brelu,dropoutDict,features,scalars,histograms