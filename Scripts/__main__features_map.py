"""Deep dive libs"""
from input_data_dive_test import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from depth_map_structure_dropout2 import create_structure

"""Core libs"""
import tensorflow as tf
import numpy as np

"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm


"""Python libs"""
import os
from optparse import OptionParser
from PIL import Image
import subprocess
import time
from ssim_tf import ssim_tf
from features_on_grid import put_features_on_grid

"""Verifying options integrity"""

#depois a gente coloca isso no config
#o segundo parametro e o numero de linhas pra mostrar
features_list=[["S1_conv1", 8],["S1_pool1", 8],["S1_pool2",12]]

if restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')

dataset = DataSetManager(path, val_path, pathGroundTruth, out_path, input_size, output_size, proportions)
global_step = tf.Variable(0, trainable=False, name="global_step")


#patch_input_size = (patch_size, patch_size, input_size[2])
#patch_output_size = (patch_size, patch_size, output_size[2])




#mask = [[[1.0*((i>=max_kernel_size//2) and (i<patch_size-max_kernel_size//2) and (j>=max_kernel_size//2) and (j<patch_size-max_kernel_size//2)) for k in range(3)] for j in range(patch_size)] for i in range(patch_size)]



#dataset = manager.read_data_sets2(n_images=n_images,n_images_validation=n_images_validation)

#x = tf.placeholder("float", shape=[None, np.prod(np.array(input_size))], name="input_image")
#y_ = tf.placeholder("float", shape=[None, np.prod(np.array(output_size))], name="output_image")

x = tf.placeholder("float", name="input_image")
y_ = tf.placeholder("float", name="output_image")

dout1 = tf.placeholder("float")
dout2 = tf.placeholder("float")
dout3 = tf.placeholder("float")
dout4 = tf.placeholder("float")

#tf_mask=tf.Variable(initial_value=mask, trainable=False, name="mask")

#initial = tf.constant(0,dtype='float32')
#loss_average_var = tf.Variable(initial, name="total_loss")

#count = tf.Variable(initial, name="count")


# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()

last_layer, l2_reg, feature_maps = create_structure(tf, x,input_size,[dout1,dout2,dout3,dout4])

#y_image = tf.reshape(y_, [-1, output_size[0], output_size[1], output_size[2]])
y_image = y_

#loss_function = tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)) + l2_reg_w * l2_reg
loss_function = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)))
#loss_function = tf.sqrt(tf.reduce_mean(tf.pow(tf.mul(tf.sub(last_layer, y_image),tf_mask),2)),name='Training')

# using the same function with a different name
loss_validation = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)),name='Validation')


loss_function_ssim = ssim_tf(tf,y_image,last_layer)

#loss_average = tf.div(tf.add(loss_average_var, loss_validation),tf.add(count,1));

#PSNR
#loss_function_psnr = tf.constant(20.0) * (tf.log(tf.div(tf.constant(1.0), tf.sqrt(MSE))) / tf.constant(2.302585093))



train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
#tf.add_to_collection('losses', loss_validation)




#loss_averages_op = _add_loss_summaries(loss_validation)

"""Creating summaries"""
#tf.image_summary('Input', tf.reshape(x, [batch_size, input_size[0], input_size[1], input_size[2]]))
tf.image_summary('Input', x)

tf.image_summary('Output', last_layer)
#tf.image_summary('GroundTruth', tf.reshape(y_, [batch_size, output_size[0], output_size[1], output_size[2]]))
tf.image_summary('GroundTruth', y_)
#with tf.variable_scope('scale_1') as scope_conv: 
# tf.get_variable_scope().reuse_variables()
# ft=tf.get_variable("Scale1_first_relu")
# tf.image_summary('Features_map', put_features_on_grid (ft, 8))
for key, l in features_list:
 tf.image_summary('Features_map_'+key, put_features_on_grid(feature_maps[key], l))
# tf.histogram_summary('InputHist', x)
# tf.histogram_summary('OutputHist', last_layer)

tf.scalar_summary('Loss', loss_function)

tf.scalar_summary('Loss_SSIM', loss_function_ssim)
tf.scalar_summary('L2_loss', l2_reg)

#tf.scalar_summary('Loss_PSNR', loss_function_psnr)
# tf.scalar_summary('learning_rate', learning_rate)

#val = tf.scalar_summary('Loss_Average', loss_average)

#with tf.variable_scope('scale_1') as scope_conv:
  
#  scope_conv.reuse_variables()
#  weights = tf.get_variable('W_S1_conv1')
  
  #features = tf.get_variable('feature1_vis')

#  grid_x = grid_y = 8   # to get a square grid for 64 conv1 features
#  gridw = put_kernels_on_grid (weights, (grid_y, grid_x))
  #gridf = put_kernels_on_grid (features, (grid_y, grid_x))
#  tf.image_summary('conv1/kernels', gridw, max_images=1)
  #tf.image_summary('conv1/Features', gridf, max_images=1)



summary_op = tf.merge_all_summaries()
saver = tf.train.Saver(tf.all_variables())

val  =tf.scalar_summary('Loss_Validation', loss_validation)


sess.run(tf.initialize_all_variables())





summary_writer = tf.train.SummaryWriter(summary_path,
                                            graph_def=sess.graph_def)

# """Open tensorboard"""
# subprocess.Popen(['gnome-terminal', '-e', 'tensorboard --logdir ' + summary_path], shell=True)
  
"""Load a previous model if restore is set to True"""

if not os.path.exists(models_path):
  os.mkdir(models_path)
ckpt = tf.train.get_checkpoint_state(models_path)

print ckpt
if restore:
  if ckpt.model_checkpoint_path:
    print 'Restoring from ', ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
  ckpt = 0


print 'Logging into ' + summary_path



"""Training"""

lowest_error = 1.5;
lowest_val  = 1.5;
lowest_iter = 1;
lowest_val_iter = 1;

#i + int(ckpt.model_checkpoint_path.split('-')[1])


if ckpt:
  initialIteration = int(ckpt.model_checkpoint_path.split('-')[1])
else:
  initialIteration = 1

for i in range(initialIteration, n_epochs*n_images_dataset):

  
  epoch_number = 1.0+ (float(i)*float(batch_size))/float(n_images_dataset)


  
  """ Do validation error and generate Images """
  #if i%(n_images/batch_size) == 1:
  #  dataset = manager.read_data_sets2(n_images=n_images,n_images_validation=n_images_validation)
  
  batch = dataset.train.next_batch(batch_size)
  



  """Save the model every 300 iterations"""
  if i%300 == 0:
    # if ckpt:
    #   saver.save(sess, models_path + 'model.ckpt', global_step=i + int(ckpt.model_checkpoint_path.split('-')[1]))
    #   print 'Model saved.'
    # else:
    saver.save(sess, models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'


  start_time = time.time()
  """Calculate the loss"""
  #train_accuracy = loss_function.eval(feed_dict={
  #    x:batch[0], y_: batch[1]})
  """Run training and write the summaries"""
  #train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],dout1:dropout[0],dout2:dropout[1],dout3:dropout[2],dout4:dropout[3]})
  


  duration = time.time() - start_time

  if i%20 == 0:
    num_examples_per_step = batch_size 
    examples_per_sec = num_examples_per_step / duration
    train_accuracy = sess.run(loss_function, feed_dict={x: batch[0], y_: batch[1],dout1:1,dout2:1,dout3:1,dout4:1})
    if  train_accuracy < lowest_error:
      lowest_error = train_accuracy
      lowest_iter = i
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"%(epoch_number, i, i*batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))
  


  #if ckpt:
  #  summary_writer.add_summary(summary_str, i + int(ckpt.model_checkpoint_path.split('-')[1]))
  #else:
  """ Writing summary, not at every iterations """
  if i%20 == 0:
    batch_val = dataset.validation.next_batch(batch_size)
    summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1],dout1:1,dout2:1,dout3:1,dout4:1})
    summary_str_val,result= sess.run([val,last_layer], feed_dict={x: batch_val[0], y_: batch_val[1],dout1:1,dout2:1,dout3:1,dout4:1})
    summary_writer.add_summary(summary_str,i)

    """ Check here the weights """


    result = Image.fromarray((result[0,:,:,:]*255).astype(np.uint8))
    result.save(out_path + str(str(i)+ '.jpg'))
    summary_writer.add_summary(summary_str_val,i)
  

