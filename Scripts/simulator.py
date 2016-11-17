import tensorflow as tf
import numpy as np


def acquireProperties(turbidity_patches):
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

def applyTurbidity(images, depths, c, binf, ranges):#, max_range_dev):
	trans=tf.exp(-depths*c*ranges)
	return images*trans + binf *(1-trans)

