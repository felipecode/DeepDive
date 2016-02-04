input_size = (128, 128, 3)
output_size = (128, 128, 3)

n_images = 1500  #Number of images to be generated at each time in memory

learning_rate = 1e-5
init_std_dev = 0.1

models_path = 'lowturb_dirt_or_rain_dataset_1-1/'
path = '/home/nautec/DeepDive/Simulator/Dataset1_2R1/Training/'
# path = '/home/nautec/Framework-UFOG/CPP/Ancuti3.png'
summary_path = '/tmp/deep_dive_49'
restore = True
evaluation = False

num_gpus = 3

if not evaluation:
  batch_size = 4
else:
  batch_size = 1
