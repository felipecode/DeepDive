import tensorflow as tf
import numpy as np
import math
from PIL import Image
import os



def save_optimazed_image_to_disk(opt_output, channel,n_channels,key,path):
	opt_output_rescaled = (opt_output - opt_output.min())
	opt_output_rescaled *= (255/opt_output_rescaled.max())
	im = Image.fromarray(opt_output_rescaled.astype(np.uint8))
	file_name="opt_"+str(channel).zfill(len(str(n_channels)))+".bmp"
	folder_name=path+"/feature_maps/"+key+"/optimization"
	if not os.path.exists(folder_name):
	  os.makedirs(folder_name)
	im.save(folder_name+"/"+file_name)

def save_images_to_disk(result_imgs,input_imgs,gt_imgs, path):
		result_imgs=(result_imgs * 255).round().astype(np.uint8)
		for j in xrange(result_imgs.shape[0]):
			im = Image.fromarray(result_imgs[j])
			file_name="output.bmp"
			im_folder=str(j).zfill(len(str(result_imgs.shape[0])))
			folder_name=path+"/output/"+im_folder
			if not os.path.exists(folder_name):
				os.makedirs(folder_name)
			im.save(folder_name+"/"+file_name)

		input_imgs=(input_imgs * 255).round().astype(np.uint8)
		for j in xrange(input_imgs.shape[0]):
			im = Image.fromarray(input_imgs[j])
			file_name="input.bmp"
			im_folder=str(j).zfill(len(str(input_imgs.shape[0])))
			folder_name=path+"/input/"+im_folder
			if not os.path.exists(folder_name):
				os.makedirs(folder_name)
			im.save(folder_name+"/"+file_name)

		gt_imgs=(gt_imgs * 255).round().astype(np.uint8)
		for j in xrange(gt_imgs.shape[0]):
			im = Image.fromarray(gt_imgs[j])
			file_name="ground_truth.bmp"
			im_folder=str(j).zfill(len(str(gt_imgs.shape[0])))
			folder_name=path+"/ground_truth/"+im_folder
			if not os.path.exists(folder_name):
				os.makedirs(folder_name)
			im.save(folder_name+"/"+file_name) 

def save_feature_maps_to_disk(feature_maps,feature_names,path):

	for ft, key in zip(feature_maps,feature_names):
		ft_img = (ft - ft.min())
		ft_img*=(255/ft_img.max())
		for k in xrange(ft.shape[0]):
			for l in xrange(ft.shape[3]):
				ch_img=ft_img[k,:,:,l].astype(np.uint8) 
				im = Image.fromarray(ch_img)
				file_name=str(l).zfill(len(str(ft.shape[3])))+".bmp"
				im_folder=str(k).zfill(len(str(ft.shape[0])))
				folder_name=path+"/feature_maps/"+key+"/"+im_folder
				if not os.path.exists(folder_name):
				  os.makedirs(folder_name)
				im.save(folder_name+"/"+file_name)

def put_features_on_grid_np (features, pad=4):
 iy=features.shape[1]
 ix=features.shape[2]
 n_ch=features.shape[3]
 b_size=features.shape[0]
 square_size=int(math.ceil(np.sqrt(n_ch)))
 z_pad=square_size**2-n_ch
 features = np.pad(features, [[0,0],[0,0],[0,0],[0,z_pad]], mode='constant',constant_values=0)
 features = np.reshape(features,[b_size,iy,ix,square_size,square_size])
 features = np.pad(features, [[0,0],[pad,0],[pad,0],[0,0],[0,0]], mode='constant',constant_values=0)
 iy+=pad
 ix+=pad
 features = np.transpose(features,(0,3,1,4,2))
 return np.reshape(features,[-1,square_size*iy,square_size*ix,1])

def put_kernels_on_grid_np (kernels, pad=4):
 iy=kernels.shape[0]
 ix=kernels.shape[1]
 n_ch=kernels.shape[3]
 square_size=int(math.ceil(np.sqrt(n_ch)))
 z_pad=square_size**2-n_ch
 kernels = np.pad(kernels, [[0,0],[0,0],[0,0],[0,z_pad]], mode='constant',constant_values=0)
 kernels = np.transpose(kernels,(0,1,3,2))
 kernels = np.reshape(kernels,[iy,ix,square_size,square_size,3])
 kernels = np.pad(kernels, [[pad,0],[pad,0],[0,0],[0,0],[0,0]], mode='constant',constant_values=0)
 iy+=pad
 ix+=pad
 kernels = np.transpose(kernels,(2,0,3,1,4))
 kernels = np.reshape(kernels,[square_size*iy,square_size*ix,3])
 return np.expand_dims(kernels, axis=0)

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
