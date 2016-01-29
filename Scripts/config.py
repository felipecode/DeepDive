input_size = (184, 184, 3)
output_size = (184, 184, 3)

n_images = 800  #Number of images to be generated at each time in memory

learning_rate = 1e-5
init_std_dev = 0.1

models_path = 'models_sigmoid_stddev0.1_scaling/'
path = '/home/nautec/DeepDive/Simulator/Dataset1/Training/'
# path = '/home/nautec/exemplo.jpg'
summary_path = '/tmp/deep_dive_23'
restore = True
evaluation = False


if not evaluation:
  batch_size = 2
else:
  batch_size = 1