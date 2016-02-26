"""Training Configuration"""
learning_rate = 1e-5
init_std_dev = 0.1
l2_reg_w     = 1e-4
batch_size   = 3

"""Dataset Configuration"""
input_size = (128, 128, 3)
output_size = (128, 128, 3)
path = '/home/nautec/DeepDive/Simulator/Dataset1_4/TPatches/Training/'
n_images = 1500  			#Number of images to be generated at each time in memory
n_images_dataset = 500000	#Number of images in the whole dataset
proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]	#Proportion of each folder to be loaded

"""Saving Configuration"""
models_path = 'double_7x7_15x15_3x3_5x51.4/'
# path = '/home/nautec/Framework-UFOG/CPP/ancuti4.png'
summary_path = '/tmp/dataset1_4_first_test6'

"""Execution Configuration"""
restore = True
evaluation = False

num_gpus = 3

if evaluation:
	batch_size = 1
