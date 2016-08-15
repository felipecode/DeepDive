import tensorflow as tf
import numpy as np
from config import configOptimization



def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    std = tf.sqrt(tf.reduce_mean(tf.square(img)))
    return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    k = np.float32([1,4,6,4,1])
    k = np.outer(k, k)
    k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
    img = tf.expand_dims(img,0)

    levels = []
    for i in xrange(scale_n):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
        levels.append(hi)
	img=lo
    levels.append(img)
    tlevels=levels[::-1]
    tlevels = map(normalize_std, tlevels)

    img = tlevels[0]
    for hi in tlevels[1:]:
        img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img[0,:,:,:]

def optimize_feature(input_size, x, feature_map):
 config= configOptimization()
 images = np.empty((1, input_size[0], input_size[1], input_size[2]))
 img_noise = np.random.uniform(low=0.0, high=1.0, size=input_size)
 config= configOptimization()
 #graph=sess.graph
 #x=graph.get_tensor_by_name("input_image:0")
 sess=tf.get_default_session()
 t_score = tf.reduce_mean(feature_map)
 t_grad = tf.gradients(t_score, x)[0]

 if config.lap_grad_normalization:
  grad_norm=lap_normalize(t_grad[0,:,:,:])
 else:
  grad_norm=normalize_std(t_grad)

 images[0] = img_noise.copy()
 step_size=config.opt_step
 for i in xrange(1,config.opt_n_iters+1):
  feedDict={x: images}
  g, score = sess.run([grad_norm, t_score], feed_dict=feedDict)
  # normalizing the gradient, so the same step size should work for different layers and networks
  images[0] = images[0]+g*step_size
  #l2 decay
  images[0] = images[0]*(1-config.decay)
  #gaussian blur
  if config.blur_iter:
   if i%config.blur_iter==0:
    images[0] = gaussian_filter(images[0], sigma=config.blur_width)
  #clip norm
  # norms=np.linalg.norm(images[0], axis=2, keepdims=True)
  # n_thrshld=np.sort(norms, axis=None)[int(norms.size*config.norm_pct_thrshld)]
  # images[0]=images[0]*(norms>=n_thrshld)
  # #clip contribution
  # contribs=np.sum(images[0]*g[0], axis=2, keepdims=True)
  # c_thrshld=np.sort(contribs, axis=None)[int(contribs.size*config.contrib_pct_thrshld)]
  # images[0]=images[0]*(contribs>=c_thrshld)

 return images[0].astype(np.float32)
