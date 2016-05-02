"""configuration file"""

class configMain:
	def __init__(self):
		self.learning_rate = 5*1e-3
		self.batch_size = 1
		self.n_epochs = 40   # the number of epochs that we are going to run
		self.training_path = '../datasets/dataset2_2F/Training/'
		self.training_path_ground_truth = '../datasets/dataset2_2F/GroundTruth'
		self.validation_path = '../datasets/dataset2_2F/Validation/'
		self.summary_path = '/tmp/dataset4_12'
		self.validation_path_ground_truth = '../datasets/dataset2_2F/ValidationGroundTruth/'
		self.models_path = 'models/deepdivearch0.2s_d2.2F_mar_21/'
		self.input_size = (512, 512, 3)
		self.output_size = (512, 512, 3)
		self.ground_truth_size = (512, 512, 3)
		self.restore = True
		self.dropout = [1,1,1,1]
		self.features_list=[["S1_conv1", 8],["S1_pool1", 8],["S1_pool2",12]]

class configOptimization:
	def __init__(self):
		self.opt_step = 5*1e-1
		self.opt_iter_n = 2000
		self.models_path = 'models/deepdivearch0.2s_d2.2F_mar_21/'
		self.summary_path = '/tmp/dataset4_12'
		self.input_size = (512, 512, 3)
		self.output_size = (512, 512, 3)
		self.restore = True
		self.dropout = [1,1,1,1]
		self.features_list=[["S1_conv1", 8],["S1_pool1", 8],["S1_pool2",12]]
	        self.features_opt_list=[["S1_conv1", 8],["S1_conv1", 7],["S1_conv1", 36]]
                self.l2_decay=True
		self.decay=0.001
		self.gaussian_blur=True
		self.blur_iter=500
		self.blur_width=1

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
		self.dropout = []
		self.features_list=[]
		self.evaluate_path = '/home/nautec/DeepDive/Local_results/RealImages/'
		self.evaluate_out_path ='/home/nautec/DeepDive/Local_results/RealImageTransmission/'
