input_size = (128, 128, 3)
output_size = (128, 128, 3)

n_images = 1500  #Number of images to be generated at each time in memory

learning_rate = 1e-5
init_std_dev = 0.1
l2_reg_w     = 0

models_path = 'teste2_64_channels_2_inception/'
path = '/home/nautec/DeepDive/Simulator/Dataset1_2R1/Training/'
# path = '/home/nautec/Framework-UFOG/CPP/Ancuti3.png'
summary_path = '/tmp/deep_dive_2_inception_2'
restore = False
evaluation = False

num_gpus = 3

if not evaluation:
  batch_size = 2
else:
  batch_size = 1
