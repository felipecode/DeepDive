"""configuration file"""

class configMain:
	def __init__(self):
		self.range_max=5.0
		self.range_min=0.0
		self.learning_rate = 1e-5
		self.lr_update_value = 1
		self.lr_update_period =1
		self.beta1=0.9
		self.beta2=0.999
		self.epsilon=1e-06
		self.use_locking=False
		self.batch_size = 16
		self.batch_size_val =16
		self.variable_names = []#['MSE']
		self.n_epochs = 120   # the number of epochs that we are going to run
		self.turbidity_path='../datasets/simteste/TurbidityV3'
		self.WEIGHTS_FILE = "vgg16_weights.npz"
		self.leveldb_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/'
		self.training_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Training/'
		self.training_transmission_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Transmission/'
		self.validation_transmission_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/ValidationTransmission/'
		self.training_path_ground_truth = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/GroundTruth/'
		self.validation_path = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/Validation/'
		self.validation_path_ground_truth = '/home/nautec/DeepDive-master/datasets/datasetDepthV6/ValidationGroundTruth/'
		self.summary_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/new_25_10BN/'
		self.models_path = '../modelo/'
		self.input_size = (224, 224,3)
		self.depth_size = (224, 224,1)
		self.output_size = (224, 224, 3)
		self.turbidity_size=(128,128)
		self.ground_truth_size = (224, 224, 3)
		self.restore = False
		self.dropout = [1,1,1,1]
		self.summary_writing_period = 20
		self.validation_period = 120
		self.histograms_list=[]#"W_conv1","W_conv2","W_conv3","W_conv4","W_conv5","W_conv6"]
		self.features_list=[]
		self.features_opt_list=[]
		self.opt_every_iter= 0
		self.save_features_to_disk=False
		self.save_json_summary=True
		self.save_error_transmission=False
		self.num_bins = 10
		self.use_tensorboard=True
		self.use_deconv=False


class configMainSimulator:
	def __init__(self):
		self.turbidity_size=(128,128)
		self.turbidity_path="/home/nautec/DeepDive/MistDataBase/"
		self.range_max=1.2
		self.range_min=0.8
		self.learning_rate = 1e-4
		self.lr_update_value = 1
		self.lr_update_period =1
		self.beta1=0.9
		self.beta2=0.999
		self.epsilon=1e-08
		self.use_locking=False
		self.batch_size = 32
		self.batch_size_val = 32
		self.variable_names = []#['MSE']
		self.n_epochs = 20   # the number of epochs that we are going to run
		self.WEIGHTS_FILE = "vgg16_weights.npz"
		self.leveldb_path = '/home/nautec/DeepDive/datasets/geral/'
		self.training_path = '../datasets/datasetDepthV6/Training/'
		self.training_transmission_path = '../datasets/datasetDepthV6/Transmission/'
		self.validation_transmission_path = '../datasets/datasetDepthV6/ValidationTransmission/'
		self.training_path_ground_truth = '../datasets/datasetDepthV6/GroundTruth/'
		self.validation_path = '../datasets/datasetDepthV6/Validation/'
		self.validation_path_ground_truth = '../datasets/datasetDepthV6/ValidationGroundTruth/'
		self.summary_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/summary_17_02_inception_res_BAC_normalized_feature/'
		self.models_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/model_17_02_inception_res_BAC_normalized_feature/'
		self.models_discriminator_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/model_17_02_inception_res_BAC_normalized_feature_discriminator/'
		self.input_size = (224, 224, 3)
		self.output_size = (224, 224, 3)
		self.depth_size = (224, 224, 1)
		self.restore = False
		self.use_discriminator = False
		self.restore_discriminator = False
		self.dropout = [1,1,1,1]
		self.summary_writing_period = 20
		self.validation_period = 120
		self.model_saving_period = 300
		self.histograms_list=[]#"W_conv1","W_conv2","W_conv3","W_conv4","W_conv5","W_conv6"]
		self.features_list=[]
		self.features_opt_list=[]
		self.opt_every_iter= 0
		self.save_features_to_disk=False
		self.save_json_summary=True
		self.save_error_transmission=False
		self.num_bins = 10
		self.use_tensorboard=True
		self.use_deconv=False
		self.use_depths=True
		self.invert = True
		self.rotate = True


class configMainSimulatorTest:
	def __init__(self):
		self.turbidity_size=(128,128)
		self.turbidity_path="/home/nautec/DeepDive/NewSimulator/MistDataBase/"
		self.range_max=1
		self.range_min=1
		self.learning_rate = 1e-4
		self.lr_update_value = 1
		self.lr_update_period =1
		self.beta1=0.9
		self.beta2=0.999
		self.epsilon=1e-08
		self.use_locking=False
		self.batch_size = 32
		self.batch_size_val = 32
		self.variable_names = []#['MSE']
		self.n_epochs = 20   # the number of epochs that we are going to run
		self.WEIGHTS_FILE = "vgg16_weights.npz"
		self.leveldb_path = '/home/nautec/DeepDive/datasets/geral/'
		self.training_path = '../datasets/datasetDepthV6/Training/'
		self.training_transmission_path = '../datasets/datasetDepthV6/Transmission/'
		self.validation_transmission_path = '../datasets/datasetDepthV6/ValidationTransmission/'
		self.training_path_ground_truth = '../datasets/datasetDepthV6/GroundTruth/'
		self.validation_path = '../datasets/datasetDepthV6/Validation/'
		self.validation_path_ground_truth = '../datasets/datasetDepthV6/ValidationGroundTruth/'
		self.summary_path = '../sumario/'
		self.models_path = '../modelo/'
		self.input_size = (224, 224, 3)
		self.output_size = (224, 224, 3)
		self.depth_size = (224, 224, 1)
		self.restore = True
		self.dropout = [1,1,1,1]
		self.summary_writing_period = 20
		self.validation_period = 120
		self.model_saving_period = 300
		self.histograms_list=[]#"W_conv1","W_conv2","W_conv3","W_conv4","W_conv5","W_conv6"]
		self.features_list=[]
		self.features_opt_list=[]
		self.opt_every_iter= 0
		self.save_features_to_disk=False
		self.save_json_summary=True
		self.save_error_transmission=False
		self.num_bins = 10
		self.use_tensorboard=True
		self.use_deconv=False
		self.use_depths=True

class configVisualization:
	def __init__(self):
		self.batch_size = 32
		self.leveldb_path = '/home/nautec/DeepDive-master/datasets/datasetECCVTurbid2/'
		self.training_path = '/home/nautec/DeepDive-master/datasets/datasetECCVTurbid2/Training/'
		self.training_transmission_path = '/home/nautec/DeepDive-master/datasets/datasetECCVTurbid2/Transmission/'
		self.validation_transmission_path = '/home/nautec/DeepDive-master/datasets/datasetECCVTurbid2/ValidationTransmission/'
		self.training_path_ground_truth = '/home/nautec/DeepDive-master/datasets/datasetECCVTurbid2/GroundTruth/'
		self.validation_path = '/home/nautec/DeepDive-master/datasets/datasetECCVTurbid2/Validation/'
		self.validation_path_ground_truth = '/home/nautec/DeepDive-master/datasets/datasetECCVTurbid2/ValidationGroundTruth/'
		self.summary_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e1/modelnew_07_9BN_tes/'
		self.models_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e1/modelnew_07_9BN/'
		self.input_size = (3536, 2234, 3)
		self.output_size = (3536, 2234, 3)
		self.ground_truth_size = (3536,2234, 3)
		self.dropout = [1,1,1,1]
		self.summary_writing_period = 20
		self.histograms_list=["W_B_conv1","W_B_conv2","W_B_conv3","W_B_conv4","W_B_conv5",
								"W_A_conv1","W_A_conv2","W_A_conv3","W_A_conv4","W_A_conv5",
								"W_A_conv6","W_A_conv7","W_C_conv1","W_C_conv2","W_C_conv3",
								"W_C_conv4","W_C_conv5"]
		self.features_list=["B_conv4"]
		self.features_opt_list=[["B_conv4",0]]
		self.save_features_to_disk=True
		self.save_json_summary=True
		self.save_error_transmission=False
		self.num_bins = 10
		self.use_tensorboard=True
		self.use_deconv=True

class configSimConvert:
	def __init__(self):
		self.imgs_path='/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasets/dataset_kinect/images'
		self.imgs_path2='/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasets/VOCB3DO/Dividido10/images'
		self.imgs_path3='/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasets/nyu/images'
		self.depths_path='/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasets/dataset_kinect/depths'
		self.depths_path2='/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasets/VOCB3DO/Dividido10/depths'
		self.depths_path3='/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/datasets/nyu/depths'
		self.leveldb_path='../datasets/geral2/'
		self.input_size = (224,224,3)

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
		self.output_size = (224, 224,3)


class configDehazenet:
	def __init__(self):
		self.learning_rate = 1e-5
		self.init_std_dev=0.01
		self.batch_size = 512
		self.batch_size_val = 2048
		self.n_epochs = 140   # the number of epochs that we are going to run
		self.leveldb_path = '/home/nautec/DeepDive/datasets/Dataset6_2/'
		self.training_path = '/home/nautec/DeepDive/datasets/Dataset6_2/Training/'
		self.training_transmission_path = '/home/nautec/DeepDive/datasets/Dataset6_2/Transmission/'
		self.training_path_ground_truth = '/home/nautec/DeepDive/datasets/Dataset6_2/GroundTruth/'
		self.validation_path = '/home/nautec/DeepDive/datasets/Dataset6_2/Validation/'
		self.validation_path_ground_truth = '/home/nautec/DeepDive/datasets/Dataset6_2/ValidationTransmission/'
		self.summary_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/tmp_28_9_2transmission_BN_Dataset6_2/'
		self.models_path = '/media/nautec/fcc48c1a-c797-4ba9-92c0-b93b9fc4dd0e/models_28_9_2transmission_BN_Dataset6_2/'
		self.input_size = (16, 16, 3)
		self.output_size = (1, 1)
		self.ground_truth_size = (16, 16)
		self.restore = True
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
		self.evaluate_out_path ='/home/nautec/DeepDive/Local_results/RealImageTransmission/antigo/'
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
