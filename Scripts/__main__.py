"""Deep dive libs"""
from input_data_dive import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
from depth_map_structure import create_structure

"""Core libs"""
import tensorflow as tf
import numpy as np

"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.measure import structural_similarity as ssim

"""Python libs"""
import os
from optparse import OptionParser
from PIL import Image
import subprocess
import time
from ssim_tf import ssim_tf

# """Options to add in terminal execution"""
# parser = OptionParser()
# parser.add_option("-l", "--logdir", dest="summary_path", default="/tmp/deep_dive",
#                   help="write logdir (same you use in tensorboard)", metavar="FILE")
# parser.add_option("-r", "--restore", dest="restore", default='False',
#                   help="True if restoring to a previous model")
# parser.add_option("-e", "--eval", dest="evaluation", default='False',
#                   help="True if evaluating the model")
# parser.add_option("-p", "--path", dest="path", default='/home/nautec/DeepDive/Simulator/Dataset1/Training/',
#                   help="path to training set. if eval is true, path points to a single image to be evaluated")


# def _add_loss_summaries(total_loss):
#   """Add summaries for losses in CIFAR-10 model.

#   Generates moving average for all losses and associated summaries for
#   visualizing the performance of the network.

#   Args:
#     total_loss: Total loss from loss().
#   Returns:
#     loss_averages_op: op for generating moving averages of losses.
#   """
#   # Compute the moving average of all individual losses and the total loss.
#   loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
#   losses = tf.get_collection('losses')
#   loss_averages_op = loss_averages.apply(losses + [total_loss])

#   # Attach a scalar summmary to all individual losses and the total loss; do the
#   # same for the averaged version of the losses.
#   for l in losses + [total_loss]:
#     # Name each loss as '(raw)' and name the moving average version of the loss
#     # as the original loss name.
#     tf.scalar_summary(l.op.name +' (raw)', l)
#     tf.scalar_summary(l.op.name, loss_averages.average(l))

#   return loss_averages_op


# (options, args) = parser.parse_args()

"""Verifying options integrity"""
if evaluation not in (True, False):
  raise Exception('Wrong eval option. (True or False)')
if restore not in (True, False):
  raise Exception('Wrong restore option. (True or False)')

manager = DataSetManager(path,val_path, input_size, proportions, n_images_dataset)
global_step = tf.Variable(0, trainable=False, name="global_step")


patch_input_size = (patch_size, patch_size, input_size[2])
patch_output_size = (patch_size, patch_size, output_size[2])




mask = [[[1.0*((i>=max_kernel_size//2) and (i<patch_size-max_kernel_size//2) and (j>=max_kernel_size//2) and (j<patch_size-max_kernel_size//2)) for k in range(3)] for j in range(patch_size)] for i in range(patch_size)]



if not evaluation:
  dataset = manager.read_data_sets(n_images=n_images,n_images_validation=0)

x = tf.placeholder("float", shape=[None, np.prod(np.array(input_size))], name="input_image")
y_ = tf.placeholder("float", shape=[None, np.prod(np.array(output_size))], name="output_image")
tf_mask=tf.Variable(initial_value=mask, trainable=False, name="mask")

#initial = tf.constant(0,dtype='float32')
#loss_average_var = tf.Variable(initial, name="total_loss")

#count = tf.Variable(initial, name="count")


# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()

last_layer, l2_reg = create_structure(tf, x,input_size)

y_image = tf.reshape(y_, [-1, output_size[0], output_size[1], output_size[2]])

#loss_function = tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)) + l2_reg_w * l2_reg

loss_function = tf.sqrt(tf.reduce_mean(tf.pow(tf.mul(tf.sub(last_layer, y_image),tf_mask),2)),name='Training')

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
tf.image_summary('Input', tf.reshape(x, [batch_size, input_size[0], input_size[1], input_size[2]]))
tf.image_summary('Output', last_layer)
tf.image_summary('GroundTruth', tf.reshape(y_, [batch_size, output_size[0], output_size[1], output_size[2]]))
# tf.histogram_summary('InputHist', x)
# tf.histogram_summary('OutputHist', last_layer)

tf.scalar_summary('Loss', loss_function)

tf.scalar_summary('Loss_SSIM', loss_function_ssim)
tf.scalar_summary('L2_loss', l2_reg)

#tf.scalar_summary('Loss_PSNR', loss_function_psnr)
# tf.scalar_summary('learning_rate', learning_rate)

#val = tf.scalar_summary('Loss_Average', loss_average)





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

valiter=17;
for i in range(initialIteration, n_epochs*len(manager.im_names_val)):

  
  epoch_number = 1.0+ float(i)*batch_size/float(len(manager.im_names_val))

  
  """ Do validation error and generate Images """

  # when finish epoch
  if epoch_number>valiter:
    print ' Validating'
    summary_str_val_avg=0

    for v in range(1, int(n_images_validation_dataset/(batch_size))):
      

      if v%(n_images_validation/batch_size) == 1:
        dataset = manager.read_data_sets(n_images=0,n_images_validation=n_images_validation)

      batch_val = dataset.validation.next_batch(batch_size)
  

      """Calculate the loss for the validation image and also print this image for further validation"""
      
      start_time = time.time()
      summary_str_val,train_accuracy,result = sess.run([val, loss_validation,last_layer ], feed_dict={x: batch_val[0], y_: batch_val[1]})
      duration = time.time() - start_time


      if  train_accuracy < lowest_val:
        lowest_val = train_accuracy
        lowest_val_iter = v;

      #if ckpt:
      #  print("Validation step %d, images used %d, loss %g, lowest_error %g on %d"%((i + int(ckpt.model_checkpoint_path.split('-')[1]))/500, i*batch_size, train_accuracy, lowest_val,lowest_val_iter))
      #else:
      #print  int(summary_str_val)



      if v%50 == 0:
        num_examples_per_step = batch_size 
        examples_per_sec = num_examples_per_step / duration
        print("Validation step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"%(v, v*batch_size, train_accuracy, lowest_val,lowest_val_iter,examples_per_sec))
        result = Image.fromarray((result[0,:,:,:] * 255).astype(np.uint8))
        result.save(out_path +str(int(valiter)) + '/'+ str(v) + '.jpg')

        
        summary_writer.add_summary(summary_str_val, i+v)


      #if ckpt:
      #  summary_writer.add_summary(summary_str, i + int(ckpt.model_checkpoint_path.split('-')[1]))
      #else:

      
    valiter = valiter +1
    # reload the other dataset.
    dataset = manager.read_data_sets(n_images=n_images,n_images_validation=0)

  """Read dataset images again not to waste CPU"""
  


  if i%(n_images/batch_size) == 0:
    dataset = manager.read_data_sets(n_images=n_images,n_images_validation=0)
  
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

  train_accuracy,_ = sess.run([loss_function, train_step], feed_dict={x: batch[0], y_: batch[1]})
  duration = time.time() - start_time
  if  train_accuracy < lowest_error:
    lowest_error = train_accuracy
    lowest_iter = i


  if i%10 == 0:
    num_examples_per_step = batch_size 
    examples_per_sec = num_examples_per_step / duration
    train_accuracy
    print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"%(epoch_number, i, i*batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))
  

  #if ckpt:
  #  summary_writer.add_summary(summary_str, i + int(ckpt.model_checkpoint_path.split('-')[1]))
  #else:
  """ Writing summary, not at every iterations """
  if i%30 == 0:
    summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1]})
    summary_writer.add_summary(summary_str, i+ int(n_images_validation_dataset/(batch_size)*(valiter-1)))


# image =h_noise3.eval()[0]
# sumImage = image[:,:,0]
# for i in range(1,192):
#   sumImage= image[:,:,i] + sumImage

# sumImage = sumImage/192.0

# maxImage = np.amax(sumImage)
# sumImage = np.array((sumImage/maxImage)*255,dtype=np.uint8)

# print maxImage

# print sumImage

# implot = plt.imshow(sumImage,cmap= cm.Greys_r)

# plt.show()