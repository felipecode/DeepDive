import tensorflow as tf
import numpy as np
import glob
from PIL import Image

def acquireProperties(config,sess):
	t_imgs_names=glob.glob(config.turbidity_path + "/*.jpg")
	t_batch_size=len(t_imgs_names)
	turbidities=np.empty((t_batch_size,)+config.turbidity_size+(3,))
	for i in xrange(t_batch_size):
	  t_image = Image.open(t_imgs_names[i]).convert('RGB')
	  t_image = t_image.resize(config.turbidity_size, Image.ANTIALIAS)
	  t_image = np.asarray(t_image)
	  t_image = t_image.astype(np.float32)
	  turbidities[i] = np.multiply(t_image, 1.0 / 255.0)

	tf_turbidity=tf.placeholder("float",turbidities.shape, name="turbidity")
	properties=_acquireProperties(tf_turbidity)

	c, binf=sess.run(properties, feed_dict={tf_turbidity: turbidities})
	#colocando os vetores no tamanho do batch, nao sei se tem um jeito melhor de fazer isso
	c_old=c
	c=np.empty((config.batch_size,c_old.shape[1]))
	for i in xrange(config.batch_size):
	  c[i]=c_old[i%len(c_old)]
	c=np.reshape(c,[config.batch_size,1,1,3])

	binf_old=binf
	binf=np.empty((config.batch_size,binf_old.shape[1]))
	for i in xrange(config.batch_size):
	  binf[i]=binf_old[i%len(binf_old)]
	binf=np.reshape(binf,[config.batch_size,1,1,3])


	range_step=(config.range_min-config.range_max)/(t_batch_size-1)
	range_values=np.empty(t_batch_size)
	for i in xrange(t_batch_size):
	  range_values[i]=(i)*range_step+config.range_max

	#print range_values

	#parte fixa do range
	range_array=np.empty(config.batch_size)
	for i in xrange(config.batch_size):
	  range_array[i]=range_values[(i/(config.batch_size/t_batch_size))%t_batch_size]
	range_array=np.reshape(range_array,[config.batch_size, 1,1,1])
	return c,binf,range_array


def _acquireProperties(turbidity_patches):
	l=1.06      #Schechner,2006
	T=1.0       #Transmission coefficient at the water surface - from Processing of field spectroradiometric data - EARSeL
	I0=1.0
	turbidity_patches=tf.maximum(turbidity_patches,0.001)
	turbidity_patches=turbidity_patches/(l*T*I0)
	turbidity_patches=-tf.log(turbidity_patches)
	c=tf.reduce_mean(turbidity_patches, reduction_indices=(1,2))
	patch_max=tf.reduce_max(turbidity_patches, reduction_indices=(1,2))
	binf=l*T*I0*tf.exp(-c*patch_max)
	return c,binf

def applyTurbidity(images, depths, c, binf, ranges):
	trans=tf.exp(-depths*c*ranges)
	return images*trans + binf *(1-trans)

