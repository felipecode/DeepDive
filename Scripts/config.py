"""Training Configuration"""
learning_rate = 1e-5
init_std_dev = 0.1
l2_reg_w     = 1e-4
batch_size   = 3

"""Dataset Configuration"""
input_size = (128, 128, 3)
output_size = (128, 128, 3)
path = '/home/nautec/DeepDive/Simulator/Dataset2_0/Training/'
val_path = '/home/nautec/DeepDive/Simulator/Dataset2_0/Validation/'
n_images = 100  			#Number of images to be generated at each time in memory
n_images_dataset = 500000	#Number of images in the whole dataset
n_images_validation = 1500
n_images_validation_dataset = 20036

#proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]	#Proportion of each folder to be loaded
proportions = [1, 1, 1, 1, 1, 1,1]
n_epochs = 40   # the number of epochs that we are going to run

"""Saving Configuration"""
models_path = 'inceptionsval_d2.0_mar_1/'
# path = '/home/nautec/Framework-UFOG/CPP/ancuti4.png'
summary_path = '/tmp/dataset1_4_first_test45'
out_path = '/home/nautec/DeepDive/Simulator/Dataset2_0/ValidationResults/'

"""Execution Configuration"""	
restore = True
evaluation = False

num_gpus = 3

if evaluation:
	batch_size = 1
