"""Deep dive libs"""
from deep_dive import DeepDive
import input_data_dive

"""Structure"""
import sys
sys.path.append('structures')
from dirt_or_rain_structure import create_structure

"""Core libs"""
import tensorflow as tf
import numpy as np

"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""Python libs"""
from optparse import OptionParser
from PIL import Image

parser = OptionParser()
parser.add_option("-l", "--logdir", dest="summary_path", default="/tmp/deep_dive",
                  help="write logdir (same you use in tensorboard)", metavar="FILE")
parser.add_option("-r", "--restore", dest="restore", default='False',
                  help="True if restoring to a previous model")
parser.add_option("-e", "--eval", dest="evaluation", default='False',
                  help="True if evaluating the model")

(options, args) = parser.parse_args()
print 'Logging into ' + options.summary_path

input_size = (184, 184, 3)
output_size = (184, 184, 3)
n_images = 400  #Number of images reading at each time

global_step = tf.Variable(0, trainable=False, name="global_step")

if options.evaluation != 'True':
  dataset = input_data_dive.read_data_sets(path='/home/nautec/DeepDive/Simulator/Dataset1/Training/', input_size=input_size, n_images=n_images)

x = tf.placeholder("float", shape=[None, np.prod(np.array(input_size))], name="input_image")
y_ = tf.placeholder("float", shape=[None, np.prod(np.array(output_size))], name="output_image")

sess = tf.InteractiveSession()

deep_dive = DeepDive()
path = '../Local_aux/weights/'

h_conv3 = create_structure(tf, x)

batch_size = 1
learning_rate = 1e-5

y_image = tf.reshape(y_, [-1, output_size[0], output_size[1], output_size[2]])

loss_function = tf.reduce_mean(tf.pow(tf.sub(h_conv3, y_image),2))
#PSNR
# loss_function = tf.constant(20.0) * (tf.log(tf.div(tf.constant(1.0), tf.sqrt(MSE))) / tf.constant(2.302585093))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

tf.image_summary('Input', tf.reshape(x, [batch_size, input_size[0], input_size[1], input_size[2]]))
tf.image_summary('Output', h_conv3)
tf.image_summary('GroundTruth', tf.reshape(y_, [batch_size, output_size[0], output_size[1], output_size[2]]))
tf.scalar_summary('Loss', loss_function)

summary_op = tf.merge_all_summaries()

saver = tf.train.Saver(tf.all_variables())

# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

sess.run(tf.initialize_all_variables())

summary_writer = tf.train.SummaryWriter(options.summary_path,
                                            graph_def=sess.graph_def)


ckpt = tf.train.get_checkpoint_state('models')
if ckpt and ckpt.model_checkpoint_path and options.restore == 'True':
  print 'Restoring from ', ckpt.model_checkpoint_path  
  saver.restore(sess, 'models/' + ckpt.model_checkpoint_path)

if options.evaluation == 'True':

  path = '/home/nautec/Downloads/TURBID/Photo3D/a16pd.jpg'
  im = Image.open(path).convert('RGB')

  """TODO - arrumar erro de resize e trocar para chunks"""
  im = im.resize((184, 184))
  im = np.array(im, dtype=np.float32)

  im = im.reshape([1, np.prod(np.array(input_size))])
  im = im.astype(np.float32)
  im = np.multiply(im, 1.0 / 255.0)

  out = np.zeros([output_size[0], output_size[1], output_size[2]]).reshape([1, np.prod(np.array(output_size))])
  result = sess.run(h_conv3, feed_dict={x: im, y_: out})
  summary_str = sess.run(summary_op, feed_dict={x: im, y_: out})
  summary_writer.add_summary(summary_str, 1)
  
  fig = plt.figure()
  fig.add_subplot(1,2,1)
  plt.imshow(im.reshape([184, 184, 3]))
  fig.add_subplot(1,2,2)

  plt.imshow(result[0])
  plt.show()

  sys.exit()


for i in range(20000):
  
  if i%n_images == 0:
    dataset = input_data_dive.read_data_sets(path='/home/nautec/DeepDive/Simulator/Dataset1/Training/', input_size=input_size, n_images=n_images)
  
  batch = dataset.train.next_batch(batch_size)
  if i%300 == 0:
    saver.save(sess, 'models/model.ckpt', global_step=i)
    print 'Model saved.'

  train_accuracy = loss_function.eval(feed_dict={
      x:batch[0], y_: batch[1]})
  
  if i%10 == 0:
    print("step %d, images used %d, loss %g"%(i, i*batch_size, train_accuracy))
  
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1]})
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