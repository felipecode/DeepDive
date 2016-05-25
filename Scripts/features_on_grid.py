import tensorflow as tf
import numpy as np
import math

def put_features_on_grid_np (features, pad=4):
 iy=features.shape[1]
 ix=features.shape[2]
 n_ft=features.shape[3]
 b_size=features.shape[0]
 square_size=int(math.ceil(np.sqrt(n_ft)))
 z_pad=square_size**2-n_ft
 features = np.pad(features, [[0,0],[0,0],[0,0],[0,z_pad]], mode='constant',constant_values=0)
 features = np.reshape(features,[b_size,iy,ix,square_size,square_size])
 features = np.pad(features, [[0,0],[pad,0],[pad,0],[0,0],[0,0]], mode='constant',constant_values=0)
 iy+=pad
 ix+=pad
 features = np.transpose(features,(0,3,1,4,2))
 return np.reshape(features,[-1,square_size*iy,square_size*ix,1])

def put_features_on_grid_tf (features, cy=1, pad=4):
 iy=tf.shape(features)[1]
 ix=tf.shape(features)[2]
 b_size=tf.shape(features)[0]
 features = tf.reshape(features,tf.pack([b_size,iy,ix,cy,-1]))
 cx=tf.shape(features)[4]
 features = tf.pad(features, tf.constant( [[0,0],[pad,0],[pad,0],[0,0],[0,0]]))
 iy+=pad
 ix+=pad
 features = tf.transpose(features,(0,3,1,4,2))
 return tf.reshape(features,tf.pack([-1,cy*iy,cx*ix,1]))
