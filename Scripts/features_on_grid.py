import tensorflow as tf

def put_features_on_grid (features, cx, pad=4):
 iy=tf.shape(features)[1]
 ix=tf.shape(features)[2]
 b_size=tf.shape(features)[0]
 features = tf.reshape(features,tf.pack([b_size,iy,ix,-1,cx]))
 cy=tf.shape(features)[3]
 features = tf.pad(features, tf.constant( [[0,0],[pad,0],[pad,0],[0,0],[0,0]]))
 iy+=pad
 ix+=pad
 features = tf.transpose(features,(0,3,1,4,2))
 return tf.reshape(features,tf.pack([-1,cy*iy,cx*ix,1]))
