
"""configuration file"""

class configMain:
	def __init__(self):
		self.learning_rate = 1e-5
		self.lr_update_value = 5
		self.lr_update_period = 20
		self.beta1=0.9
		self.beta2=0.999
		self.epsilon=1e-08
		self.use_locking=False
		self.batch_size = 32
		self.batch_size_val = 96
		self.variable_names = []#['MSE']
		self.n_epochs = 240   # the number of epochs that we are going to run
		self.leveldb_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/'
		self.training_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Training/'
		self.training_transmission_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Transmission/'
		self.validation_transmission_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/ValidationTransmission/'
		self.training_path_ground_truth = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/GroundTruth/'
		self.validation_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Validation/'
		self.summary_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e1/modelnew_07_9BN/'
		self.validation_path_ground_truth = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/ValidationGroundTruth/'
		self.models_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e1/modelnew_07_9BN/'
		self.input_size = (224, 224, 3)
		self.output_size = (224, 224, 3)
		self.ground_truth_size = (224,224, 3)
		self.restore = True
		self.dropout = [1,1,1,1]
		self.summary_writing_period = 20
		self.validation_period = 120
		self.histograms_list=["W_B_conv1","W_B_conv2","W_B_conv3","W_B_conv4","W_B_conv5",
								"W_A_conv1","W_A_conv2","W_A_conv3","W_A_conv4","W_A_conv5",
								"W_A_conv6","W_A_conv7","W_C_conv1","W_C_conv2","W_C_conv3",
								"W_C_conv4","W_C_conv5"]
		self.features_list=[]
		self.features_opt_list=[]
		self.opt_every_iter= 0
		self.save_features_to_disk=False
		self.save_json_summary=True
		self.save_error_transmission=False
		self.num_bins = 10
		self.use_tensorboard=True
		self.use_deconv=False

class configConvert:
	def __init__(self):
		self.leveldb_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/'
		self.training_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Training/'
		self.training_transmission_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Transmission/'
		self.validation_transmission_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/ValidationTransmission/'
		self.training_path_ground_truth = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/GroundTruth/'
		self.validation_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Validation/'
		self.validation_path_ground_truth = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/ValidationGroundTruth/'
		self.input_size = (224, 224, 3)
		self.output_size = (224, 224, 3)
		self.ground_truth_size = (224,224, 3)


class configDehazenet:
	def __init__(self):
		self.learning_rate = 1e-7
		self.init_std_dev=0.01
		self.batch_size = 32
		self.batch_size_val = 2048
		self.n_epochs = 120   # the number of epochs that we are going to run
		self.leveldb_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasetDepth16-2/'
		self.training_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasetDepth16-2/Training/'
		self.training_transmission_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasetDepth16-2/Transmission/'
		self.training_path_ground_truth = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasetDepth16-2/GroundTruth/'
		self.validation_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasetDepth16-2/Validation/'
		self.summary_path = '/tmp/22_8_transmission/'
		self.validation_path_ground_truth = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasetDepth16-2/ValidationTransmission/'
		self.models_path = 'models/22_8_transmission/'
		self.input_size = (16, 16, 3)
		self.output_size = (16, 16)
		self.ground_truth_size = (16, 16)
		self.restore = False
		self.dropout = []
		self.features_list=[]#["conv1",4],["conv2",4],["conv3",8],["conv4",8],["pool1",8], ["pool2", 8]]
		#self.features_list=[]#["conv1_1", 4], ["conv1_2", 4], ["conv1_3", 4], ["conv1_4", 4],
							#["pool1_1",8], ["pool1_2",8], ["pool1_3",8], ["pool1_4",8], ["pool2", 8],
							#["incep1_3_3", 4], ["incep1_5_5", 4], ["incep1_7_7", 4]]
		self.histograms_list=[]#"W_conv1","b_conv1","W_conv2","b_conv2","W_conv3", "b_conv3", "W_conv4", "b_conv4"]
		#self.histograms_list=[]#"W_conv1_1", "W_conv1_2", "W_conv1_3", "W_conv1_4", "b_conv1_1",
							  #"b_conv1_2", "b_conv1_3", "b_conv1_4", "W_incep1_3_3","b_incep1_3_3",
							  #"W_incep1_5_5", "b_incep1_5_5", "W_incep1_7_7", "b_incep1_7_7",
							  #"W_conv_2", "b_conv_2"]
		self.evaluate_path = '/home/nautec/DeepDive/Local_results/'
		self.evaluate_out_path ='/home/nautec/DeepDive/Local_results/RealImageTransmission/output_paper/'
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
