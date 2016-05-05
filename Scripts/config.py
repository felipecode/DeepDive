"""configuration file"""

class configMain:
	def __init__(self):
		self.learning_rate = 5*1e-3
		self.batch_size = 4
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
		self.opt_step = 1
		self.opt_iter_n = 100
		self.models_path = 'models/deepdivearch0.2s_d2.2F_mar_21/'
		self.summary_path = '/tmp/dataset4_12'
		self.input_size = (512, 512, 3)
		self.output_size = (512, 512, 3)
		self.restore = True
		self.dropout = [1,1,1,1]
		self.features_list=[["S1_conv1", 8],["S1_pool1", 8],["S1_pool2",12]]
	        self.features_opt_list=[["S1_conv1", 8],["S1_conv1", 7],["S1_conv1", 36]]
                self.l2_decay=False
		self.decay=0.001
		self.gaussian_blur=False
		self.blur_iter=10
		self.blur_width=1
		self.clip_norm=False
		self.norm_pct_thrshld=0.5
		self.clip_contrib=False
		self.contrib_pct_thrshld=0.5

class configDehazeNet:
	def __init__(self):
		self.learning_rate = 5*1e-6
		self.init_std_dev=0.01
		self.batch_size = 5
		self.n_epochs = 80   # the number of epochs that we are going to run
		self.training_path = '../datasets/dataset4_2/Training'
		self.training_path_ground_truth = '../datasets/dataset4_2/Transmission'
		self.validation_path = '../datasets/dataset4_2/Validation'
		self.summary_path = '/tmp/dataset43q'
		self.validation_path_ground_truth = '../datasets/dataset4_2/ValidationTransmission/'
		self.models_path = 'models/'
		self.input_size = (16, 16, 3)
		self.output_size = (1, 1)
		self.ground_truth_size = (16,16)
		self.restore = False
		self.dropout = []
		self.features_list=[["conv1",2],["conv2_1",1],["incep1_3_3",4],["incep1_5_5",4],["incep1_7_7",4]]
		self.histograms_list=["W_conv1","b_conv1","W_incep1_3_3","W_incep1_5_5","W_incep1_7_7"]
		self.evaluate_path = '/home/nautec/DeepDive/Local_results/RealImages/'
		self.evaluate_out_path ='/home/nautec/DeepDive/Local_results/RealImageTransmission/'
