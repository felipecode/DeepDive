"""configuration file"""

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
