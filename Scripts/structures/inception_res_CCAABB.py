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




  C_conv1 = inception_res_C(tf,x_image,3, [32,32,32,32] ,'C1')

  C_relu1 = tf.nn.relu(C_conv1+ x_image)
  mu,sigma = tf.nn.moments(C_relu1,[0,1,2])
  C_relu1 = tf.nn.batch_normalization(C_relu1,mu,sigma,None,None,0.01)



  C_conv2 = inception_res_C(tf,C_conv1,3, [32,32,32,32] ,'C2')

  C_relu2 = tf.nn.relu(C_conv2+ C_relu1)
  mu,sigma = tf.nn.moments(C_relu2,[0,1,2])
  C_relu2 = tf.nn.batch_normalization(C_relu2,mu,sigma,None,None,0.01)



  A_conv1,feat_A,hist_A = inception_res_A(tf,C_relu2,3,[6,6,6,8,12,16],'A1')
  features.update(feat_A)
  histograms.update(hist_A)

  A_relu1 = tf.nn.relu(A_conv1+ C_relu2)
  mu,sigma = tf.nn.moments(A_relu1,[0,1,2])
  A_relu1 = tf.nn.batch_normalization(A_relu1,mu,sigma,None,None,0.01)


  A_conv2,feat_A,hist_A = inception_res_A(tf,A_relu1,3,[6,6,6,8,12,16],'A2')
  features.update(feat_A)
  histograms.update(hist_A)

  A_relu2 = tf.nn.relu(A_conv2+ A_relu1)
  mu,sigma = tf.nn.moments(A_relu2,[0,1,2])
  A_relu2 = tf.nn.batch_normalization(A_relu2,mu,sigma,None,None,0.01)





  B_conv1,feat_B,hist_b = inception_res_B(tf,A_relu2,3,[24,24,28,32],'B1')
  features.update(feat_B)
  histograms.update(hist_b)

  B_relu1 = tf.nn.relu(B_conv1+ A_relu2)
  mu,sigma = tf.nn.moments(B_relu1,[0,1,2])
  B_relu1 = tf.nn.batch_normalization(B_relu1,mu,sigma,None,None,0.01)

  B_conv2,feat_B,hist_b = inception_res_B(tf,B_relu1,3,[24,24,28,32],'B2')
  features.update(feat_B)
  histograms.update(hist_b)

  #B_relu2 = tf.nn.relu(B_conv2+ B_relu1)







  return B_conv2,dropoutDict,features,scalars,histograms