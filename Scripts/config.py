
"""configuration file"""

class configMain:
	def __init__(self):
		self.learning_rate = 1e-4
		self.beta1=0.9
		self.beta2=0.999
		self.epsilon=1e-08
		self.use_locking=False
		self.batch_size = 32
		self.batch_size_val = 96
		self.variable_names = ['MSE']
		self.n_epochs = 80   # the number of epochs that we are going to run
		self.leveldb_path = '../../DeepDive-master/datasets/datasetDepthV4/'
		self.training_path = '../../DeepDive-master/datasets/datasetDepthV4/Training/'
		self.training_transmission_path = '../../DeepDive-master/datasets/datasetDepthV4/Transmission/'
		self.validation_transmission_path = '../../DeepDive-master/datasets/datasetDepthV4/ValidationTransmission/'
		self.training_path_ground_truth = '../../DeepDive-master/datasets/datasetDepthV4/GroundTruth'
		self.validation_path = '../../DeepDive-master/datasets/datasetDepthV4/Validation/'
		self.summary_path = '/tmp/new_1_8/'
		self.validation_path_ground_truth = '../../DeepDive-master/datasets/datasetDepthV4/ValidationGroundTruth/'
		self.models_path = 'models/new_1_8/'
		self.input_size = (224, 224, 3)
		self.output_size = (224, 224, 3)
		self.ground_truth_size = (224,224, 3)
		self.restore = True
		self.dropout = [1,1,1,1]
		self.summary_writing_period = 4
		self.validation_period = 120
		self.histograms_list=["W_B_conv1","b_B_conv1","W_B_conv2","b_B_conv2","W_B_conv3","b_B_conv3","W_B_conv4","b_B_conv4","W_B_conv5","b_B_conv5",
								"W_A_conv1","b_A_conv1","W_A_conv2","b_A_conv2","W_A_conv3","b_A_conv3","W_A_conv4","b_A_conv4","W_A_conv5","b_A_conv5",
								"W_A_conv6","b_A_conv6","W_A_conv7","b_A_conv7","W_C_conv1","b_C_conv1","W_C_conv2","b_C_conv2","W_C_conv3","b_C_conv3",
								"W_C_conv4","b_C_conv4","W_C_conv5","b_C_conv5"]
		self.features_list=[]#"S1_conv1","S1_pool1","S1_pool2","S3_incep1"]
		self.features_opt_list=[]#["S1_conv1", 0],["S1_conv1", 63],["S3_incep1",0]]
		self.opt_every_iter=0
		self.save_features_to_disk=True
		self.save_json_summary=True
		self.save_error_transmission=True
		self.use_tensorboard=True

class configDehazenet:
	def __init__(self):
		self.learning_rate = 5*1e-7
		self.init_std_dev=0.01
		self.batch_size = 5
		self.batch_size_val = 256
		self.n_epochs = 80   # the number of epochs that we are going to run
		self.training_path = '../datasets/UDataset16x16/Training'
		self.training_path_ground_truth = '../datasets/UDataset16x16/Transmission'
		self.validation_path = '../datasets/UDataset16x16/Validation'
		self.summary_path = '/tmp/transmission_2_8'
		self.validation_path_ground_truth = '../datasets/UDataset16x16/ValidationTransmission/'
		self.models_path = 'models/transmission_2_8/'
		self.input_size = (16, 16, 3)
		self.output_size = (16, 16)
		self.ground_truth_size = (16, 16)
		self.restore = False
		self.dropout = []
		self.features_list=[["conv1",4],["conv2",4],["conv3",8],["conv4",8],["pool1",8],["conv5",1]]
		self.histograms_list=["W_conv1","b_conv1","W_conv2","b_conv2","W_conv3", "b_conv3", "W_conv4", "b_conv4", "W_conv5", "b_conv5"]
		self.evaluate_path = '/home/nautec/DeepDive/Local_results/'
		self.evaluate_out_path ='/home/nautec/DeepDive/Local_results/RealImageTransmission/'
		self.opt_every_iter=100

class configOptimization:
	def __init__(self):
		self.opt_step=1
		self.opt_n_iters=50
		self.decay=0
		self.blur_iter=0
		self.blur_width=1
		self.norm_pct_thrshld=0
		self.contrib_pct_thrshld=0
		self.lap_grad_normalization=True
