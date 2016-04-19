"""
learning_rate = 5*1e-5
init_std_dev = 0.1
l2_reg_w     = 1e-4
batch_size   = 4
max_kernel_size = 15

#patch_size = 128

#array_path = '../Dataset2_2F/Arrays/'
input_size = (512, 512, 3)
output_size = (512, 512, 3)
path = '../Dataset2_2F/Training/'
val_path = '../Dataset2_2F/Validation/'
#n_images = 200			#Number of images to be generated at each time in memory
n_images_dataset = 28851	#Number of images in the whole dataset
#n_images_validation = 20
n_images_validation_dataset = 3507

#proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]	#Proportion of each folder to be loaded
proportions = [1, 1, 1, 1, 1, 1,1]
n_epochs = 40   # the number of epochs that we are going to run

models_path = 'deepdivearch0.2s_d2.2F_mar_21/'
# path = '/home/nautec/Framework-UFOG/CPP/ancuti4.png'
summary_path = '/tmp/dataset3_0_12'
out_path = '../Dataset2_2F/ValidationResults/'

restore = False


num_gpus = 2




dropout = []
dropout.append(1)
dropout.append(1)
dropout.append(1)
dropout.append(1) """


class configMain:
	def __init__(self):
		self.learning_rate = 5*1e-5
		self.batch_size = 4
		self.n_epochs = 40   # the number of epochs that we are going to run
		self.training_path = '../datasets/dataset2_2F/Training/'
		self.training_path_ground_truth = '../datasets/dataset2_2F/GroundTruth'
		self.validation_path = '../datasets/dataset2_2F/Validation/'
		self.summary_path = '/tmp/dataset4_124'
		self.validation_path_ground_truth = '../datasets/dataset4_1/ValidationGroundTruth/'
		self.models_path = 'models/deepdivearch0.2s_d2.2F_mar_21/'
		self.input_size = (512, 512, 3)
		self.output_size = (512, 512, 3)
		self.ground_truth_size = (512, 512, 3)
		self.restore = False
		self.proportions = [1, 1, 1, 1, 1, 1,1]
		self.dropout = [1,1,1,1]

class configDehazeNet:
	def __init__(self):
		self.learning_rate = 5*1e-5
		self.batch_size = 5
		self.n_epochs = 40   # the number of epochs that we are going to run
		self.training_path = '../datasets/dataset4_1/Training'
		self.training_path_ground_truth = '../datasets/dataset4_1/Transmission'
		self.validation_path = '../datasets/dataset4_1/Validation'
		self.summary_path = '/tmp/dataset4_124'
		self.validation_path_ground_truth = '../datasets/dataset4_1/ValidationTransmission/'
		self.models_path = 'models/'
		self.input_size = (16, 16, 3)
		self.output_size = (16, 16)
		self.ground_truth_size = (16,16)
		self.restore = False
		self.proportions = [1, 1, 1, 1, 1, 1,1]
