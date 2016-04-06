"""Training Configuration"""
learning_rate = 5*1e-5
init_std_dev = 0.1
l2_reg_w     = 1e-4
batch_size   = 5
max_kernel_size = 15

#patch_size = 128

"""Dataset Configuration"""
#array_path = '../datasets/dataset4_1/Arrays/'
input_size = (16, 16, 3)
output_size = (16, 16, 1)
path = '../datasets/dataset4_1/Training/'
val_path = '../datasets/dataset4_1/Validation/'
#n_images = 200			#Number of images to be generated at each time in memory
n_images_dataset = 16025	#Number of images in the whole dataset
#n_images_validation = 20
n_images_validation_dataset = 1994

#proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]	#Proportion of each folder to be loaded
proportions = [1, 1, 1, 1, 1, 1,1]
n_epochs = 40   # the number of epochs that we are going to run

"""Saving Configuration"""
models_path = 'models/'
# path = '/home/nautec/Framework-UFOG/CPP/ancuti4.png'
summary_path = '/tmp/dataset4_1'
out_path = '../datasets/dataset4_1/ValidationResults/'

"""Execution Configuration"""	
restore = False

num_gpus = 1
