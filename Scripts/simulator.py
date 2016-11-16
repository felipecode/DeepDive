import tensorflow as tf
import numpy as np


def acquireProperties(turbidity_patches):
	l=1.06      #Schechner,2006
	T=1       #Transmission coefficient at the water surface - from Processing of field spectroradiometric data - EARSeL
	I0=1.0
	turbidity_patches=tf.maximum(turbidity_patches,0.001)
	turbidity_patches=tf.truediv(turbidity_patches,(l*T*I0))
	turbidity_patches=-tf.log(turbidity_patches)
	c=tf.reduce_mean(turbidity_patches, reduction_indices=(1,2))
	patch_max=tf.reduce_max(turbidity_patches, reduction_indices=(1,2))
	binf=l*T*I0*tf.exp(tf.mul(-c,patch_max))
	return c,binf

def applyTurbidity(images, depths, c, binf, ranges):
	max_range_dev=ranges[0]/2.0
	ranges+=tf.random_uniform(ranges.get_shape(), minval=-max_range_dev, maxval=max_range_dev) #adicionando variacao no ranges
	ranges=tf.expand_dims(tf.expand_dims(ranges,1),1)
	depths=tf.mul(depths, ranges)
	depths=tf.expand_dims(depths,3)
	trans=tf.concat(3, [tf.concat(3, [depths, depths]),depths]) #nao sei se tem um jeito melhor de fazer isso
	c=tf.expand_dims(tf.expand_dims(c,1),1)#outra gambiarra, talves tenha um jeito melhor de fazer isso
	binf=tf.expand_dims(tf.expand_dims(binf,1),1)#outra gambiarra, talves tenha um jeito melhor de fazer isso
	trans=tf.exp(-tf.mul(trans,c))
	Ed=tf.mul(images,trans)
	Eb=binf-tf.mul(trans,binf)
	return Ed+Eb
