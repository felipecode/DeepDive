"""Training Configuration"""
learning_rate = 1e-4
init_std_dev = 0.1
l2_reg_w     = 1e-4
batch_size   = 4
max_kernel_size = 15
max_image_size_x = 500
max_image_size_y = 500 
#patch_size = 128

"""Dataset Configuration"""
input_size = (256, 256, 3)
output_size = (256, 256, 3)
path = '/home/nautec/DeepDive/Dataset3_0/Training/'
val_path = '/home/nautec/DeepDive/Dataset3_0/Validation/'
n_images = 589			#Number of images to be generated at each time in memory
n_images_dataset = 589	#Number of images in the whole dataset
n_images_validation = 69
n_images_validation_dataset = 69

#proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]	#Proportion of each folder to be loaded
proportions = [1, 1, 1, 1, 1, 1,1]
n_epochs = 40   # the number of epochs that we are going to run

"""Saving Configuration"""
models_path = 'deepdivearchsuperbatch_d2.1_mar_18_mgpu/'
# path = '/home/nautec/Framework-UFOG/CPP/ancuti4.png'
summary_path = '/tmp/dataset2_1_dropout6'
out_path = '/home/nautec/DeepDive/Dataset2_1/ValidationResults/'

"""Execution Configuration"""	
restore = False
evaluation = False

num_gpus = 2

if evaluation:
	batch_size = 1


""" When using dropout this is useful """
dropout = []
dropout.append(0.8)
dropout.append(0.8)
dropout.append(0.8)
dropout.append(0.5) 

