"""Training Configuration"""
learning_rate = 5*1e-5
init_std_dev = 0.1
l2_reg_w     = 1e-4
batch_size   = 4
max_kernel_size = 15

#patch_size = 128

"""Dataset Configuration"""
#array_path = '../Datasets/Dataset2_2F/Arrays/'
input_size = (512, 512, 3)
output_size = (512, 512, 3)
path = '../Datasets/Dataset2_2F/Training/'
val_path = '../Datasets/Dataset2_2F/Validation/'
#n_images = 200			#Number of images to be generated at each time in memory
n_images_dataset = 28851	#Number of images in the whole dataset
#n_images_validation = 20
n_images_validation_dataset = 3507

#proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]	#Proportion of each folder to be loaded
proportions = [1, 1, 1, 1, 1, 1,1]
n_epochs = 40   # the number of epochs that we are going to run

"""Saving Configuration"""
models_path = 'deepdivearch0.2s_d2.2F_mar_21/'
# path = '/home/nautec/Framework-UFOG/CPP/ancuti4.png'
summary_path = '/tmp/dataset3_0_12'
out_path = '../datasets/Dataset2_2F/ValidationResults/'

"""Execution Configuration"""	
restore = False


num_gpus = 1




""" When using dropout, this is useful """
dropout = []
dropout.append(1)
dropout.append(1)
dropout.append(1)
dropout.append(1) 

