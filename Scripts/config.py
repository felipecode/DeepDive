"""configuration file"""

class configMain:
	def __init__(self):
		self.learning_rate = 5*1e-5
		self.batch_size = 4
		self.n_epochs = 40   # the number of epochs that we are going to run
		self.training_path = '../datasets/dataset2_2F/Training/'
		self.training_path_ground_truth = '../datasets/dataset2_2F/GroundTruth'
		self.validation_path = '../datasets/dataset2_2F/Validation/'
		self.summary_path = '/tmp/dataset4_12'
		self.validation_path_ground_truth = '../datasets/dataset4_1/ValidationGroundTruth/'
		self.models_path = 'models/deepdivearch0.2s_d2.2F_mar_21/'
		self.input_size = (512, 512, 3)
		self.output_size = (512, 512, 3)
		self.ground_truth_size = (512, 512, 3)
		self.restore = False
		self.dropout = [1,1,1,1]
		self.features_list=[["S1_conv1", 8],["S1_pool1", 8],["S1_pool2",12]]

class configDehazeNet:
	def __init__(self):
		self.learning_rate = 5*1e-6
		self.init_std_dev=0.01
		self.batch_size = 5
		self.n_epochs = 80   # the number of epochs that we are going to run
		self.training_path = '../datasets/dataset4_2/Training'
		self.training_path_ground_truth = '../datasets/dataset4_2/Transmission'
		self.validation_path = '../datasets/dataset4_2/Validation'
		self.summary_path = '/tmp/dataset43p'
		self.validation_path_ground_truth = '../datasets/dataset4_2/ValidationTransmission/'
		self.models_path = 'models/'
		self.input_size = (16, 16, 3)
		self.output_size = (1, 1)
		self.ground_truth_size = (16,16)
		self.restore = False
		self.dropout = []
		self.features_list=[["conv1_1",2],["conv1_2",2],["conv1_3",2],["conv1_4",2],["pool1_1",1],
		["pool1_2",1],["pool1_3",1],["pool1_4",1],["incep1_3_3",4],["incep1_5_5",4],["incep1_7_7",4]]
		self.histograms_list=["W_conv1_1","W_conv1_2","W_conv1_3","W_conv1_4","b_conv1_1","b_conv1_2","b_conv1_3","b_conv1_4","W_incep1_3_3","W_incep1_5_5","W_incep1_7_7"]
		self.evaluate_path = '/home/nautec/DeepDive/Local_results/RealImages/'
		self.evaluate_out_path ='/home/nautec/DeepDive/Local_results/RealImageTransmission/'
