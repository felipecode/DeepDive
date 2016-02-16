"""Deep dive libs"""
from input_data_dive import DataSetManager
from config import *

"""Structure"""
import sys
sys.path.append('structures')
from deep_dive_test_structure import create_structure

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

manager = DataSetManager(path, input_size, proportions, n_images_dataset)
global_step = tf.Variable(0, trainable=False, name="global_step")

if not evaluation:
  dataset = manager.read_data_sets(n_images=n_images)

x = tf.placeholder("float", shape=[None, np.prod(np.array(input_size))], name="input_image")
y_ = tf.placeholder("float", shape=[None, np.prod(np.array(output_size))], name="output_image")

# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()

last_layer, l2_reg = create_structure(tf, x,input_size)

y_image = tf.reshape(y_, [-1, output_size[0], output_size[1], output_size[2]])

loss_function = tf.reduce_mean(tf.pow(tf.sub(last_layer, y_image),2)) + l2_reg_w * l2_reg
#PSNR
#loss_function_psnr = tf.constant(20.0) * (tf.log(tf.div(tf.constant(1.0), tf.sqrt(MSE))) / tf.constant(2.302585093))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

"""Creating summaries"""
tf.image_summary('Input', tf.reshape(x, [batch_size, input_size[0], input_size[1], input_size[2]]))
tf.image_summary('Output', last_layer)
tf.image_summary('GroundTruth', tf.reshape(y_, [batch_size, output_size[0], output_size[1], output_size[2]]))
# tf.histogram_summary('InputHist', x)
# tf.histogram_summary('OutputHist', last_layer)
# tf.histogram_summary('GroundTruthHist', y_)
tf.scalar_summary('Loss', loss_function)
tf.scalar_summary('L2_loss', l2_reg)
#tf.scalar_summary('Loss_PSNR', loss_function_psnr)
# tf.scalar_summary('learning_rate', learning_rate)

summary_op = tf.merge_all_summaries()
saver = tf.train.Saver(tf.all_variables())

sess.run(tf.initialize_all_variables())

summary_writer = tf.train.SummaryWriter(summary_path,
                                            graph_def=sess.graph_def)

# """Open tensorboard"""
# subprocess.Popen(['gnome-terminal', '-e', 'tensorboard --logdir ' + summary_path], shell=True)
  
"""Load a previous model if restore is set to True"""
if not os.path.exists(models_path):
  os.mkdir(models_path)
ckpt = tf.train.get_checkpoint_state(models_path)
if ckpt and ckpt.model_checkpoint_path and restore:
  print 'Restoring from ', ckpt.model_checkpoint_path  
  saver.restore(sess, models_path + ckpt.model_checkpoint_path)


print 'Logging into ' + summary_path

"""Evaluation"""
if evaluation:

  overlap_size = (12, 12)

  path = path
  
  im = Image.open(path).convert('RGB')
  im = np.array(im, dtype=np.float32)
  visualizer = im

  im = np.lib.pad(im, ((input_size[0]-overlap_size[0], 0), (input_size[1]-overlap_size[1], 0), (0,0)), mode='constant', constant_values=1)

  original = im
  original = original.astype(np.float32)
  original = np.multiply(original, 1.0 / 255.0)

  height, width = im.shape[0], im.shape[1]

  united_images = np.zeros((im.shape[0]+input_size[0], im.shape[1]+input_size[1], 3), dtype=np.float32)
  out = np.zeros([output_size[0], output_size[1], output_size[2]]).reshape([1, np.prod(np.array(output_size))])

  """Separating the image in chunks"""
  for h in range(0, height, input_size[0]-(overlap_size[0]*2)):
    for w in range(0, width, input_size[1]-(overlap_size[1]*2)):
      h_end = h + input_size[0]
      w_end = w + input_size[1]

      chunk = original[h:h_end, w:w_end]
      if chunk.shape != input_size:
        chunk = np.lib.pad(chunk, ((0, input_size[0]-chunk.shape[0]), (0, input_size[1]-chunk.shape[1]), (0,0)), mode='constant', constant_values=1)

      im = chunk.reshape([1, np.prod(np.array(input_size))])

      result = sess.run(last_layer, feed_dict={x: im, y_: out})
      summary_str = sess.run(summary_op, feed_dict={x: im, y_: out})
      summary_writer.add_summary(summary_str, 1)

      united_images[h:h+input_size[0]-(overlap_size[0]*2), w:w+input_size[1]-(overlap_size[1]*2), :] = result[0][overlap_size[0]:input_size[0]-overlap_size[0], overlap_size[1]:input_size[1]-overlap_size[1]]

  jump = (input_size[0]-(2*overlap_size[0]), input_size[1]-(2*overlap_size[1]))
  united_images = united_images[jump[0]:visualizer.shape[0]+jump[0], jump[1]:visualizer.shape[1]+jump[1], :]

  fig = plt.figure()
  fig.add_subplot(1,2,1)
  plt.imshow(np.array(visualizer, dtype=np.uint8))
  fig.add_subplot(1,2,2)

  plt.imshow(united_images)
  plt.show()

  sys.exit()


"""Training"""
for i in range(1, 5000000):

  """Read dataset images again not to waste CPU"""
  if i%(n_images/batch_size) == 0:
    dataset = manager.read_data_sets(n_images=n_images)
  batch = dataset.train.next_batch(batch_size)

  """Save the model every 300 iterations"""
  if i%300 == 0:
    if ckpt:
      saver.save(sess, models_path + 'model.ckpt', global_step=i + int(ckpt.model_checkpoint_path.split('-')[1]))
      print 'Model saved.'
    else:
      saver.save(sess, models_path + 'model.ckpt', global_step=i)
      print 'Model saved.'


  start_time = time.time()
  """Calculate the loss"""
  #train_accuracy = loss_function.eval(feed_dict={
  #    x:batch[0], y_: batch[1]})
  """Run training and write the summaries"""
  #train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  summary_str,train_accuracy,_ = sess.run([summary_op, loss_function, train_step], feed_dict={x: batch[0], y_: batch[1]})
  duration = time.time() - start_time
  
  if i%10 == 0:
    num_examples_per_step = batch_size 
    examples_per_sec = num_examples_per_step / duration
    if ckpt:
      print("step %d, images used %d, loss %g, examples per second %f"%(i + int(ckpt.model_checkpoint_path.split('-')[1]), i*batch_size, train_accuracy, examples_per_sec))
    else:
      print("step %d, images used %d, loss %g, examples per second %f"%(i, i*batch_size, train_accuracy,examples_per_sec))
  

  if ckpt:
    summary_writer.add_summary(summary_str, i + int(ckpt.model_checkpoint_path.split('-')[1]))
  else:
    summary_writer.add_summary(summary_str, i)

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