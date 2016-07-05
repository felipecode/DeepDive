"""
Returns the last tensor of the network's structure
followed dropout and features dictionaries to be summarised
Input is tensorflow class and an input placeholder.  
"""


from inception_res_A import *

from inception_res_B import *

from inception_res_C import *



def create_structure(tf, x, input_size,dropout):
 
  """Deep dive libs"""
  from deep_dive import DeepDive
  deep_dive = DeepDive()



  dropoutDict={}

  x_image =x
  """ Scale 1 """

  features={}
  scalars={}
  histograms={}

  W_P_conv = deep_dive.weight_xavi_init([7,7,3,3], name='W_P_conv1')
  b_P_conv = deep_dive.bias_variable([3])
  P_conv  = tf.nn.relu(deep_dive.conv2d(x_image, W_P_conv,strides=[1, 1, 1, 1], padding='SAME') + b_P_conv)




 



  B_conv5,feat_B,hist_b = inception_res_B(tf,P_conv,3,[24,24,28,32],'B')
  features.update(feat_B)
  histograms.update(hist_b)

  B_relu = tf.nn.relu(B_conv5+ P_conv)
  mu,sigma = tf.nn.moments(B_relu,[0,1,2])
  B_relu = tf.nn.batch_normalization(B_relu,mu,sigma,None,None,0.01)
  features["B_relu"]=B_relu






  A_conv7,feat_A,hist_A = inception_res_A(tf,B_relu,3,[6,6,6,8,12,16],'A')


  A_relu = tf.nn.relu(A_conv7+ B_relu)

  mu,sigma = tf.nn.moments(A_relu,[0,1,2])
  A_relu = tf.nn.batch_normalization(A_relu,mu,sigma,None,None,0.01)
  
  features["A_relu"]=A_relu


  

  C_conv5 = inception_res_C(tf,A_relu,3, [32,32,32,32] ,'C')



  C_relu = C_conv5+ A_relu
  #mu,sigma = tf.nn.moments(C_relu,[0,1,2])
  #C_relu = tf.nn.batch_normalization(C_relu,mu,sigma,None,None,0.01)

  features["C_relu"]=C_relu



  return C_relu,dropoutDict,features,scalars,histograms