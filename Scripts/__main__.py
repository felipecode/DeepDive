"""Deep dive libs"""
from deep_dive import DeepDive
import input_data_dive

"""Structure"""
from dirt_or_rain_structure import create_structure

"""Core libs"""
import tensorflow as tf
import numpy as np

"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""Python libs"""
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-l", "--logdir", dest="summary_path", default="/tmp/deep_dive",
                  help="write logdir (same you use in tensorboard)", metavar="FILE")
parser.add_option("-e", "--eval", dest="evaluation", default='False',
                  help="True if evaluating the model")

(options, args) = parser.parse_args()
print 'Logging into ' + options.summary_path

input_size = (64, 64, 3)
output_size = (42, 42, 3)

global_step = tf.Variable(0, trainable=False, name="global_step")
dataset = input_data_dive.read_data_sets(path='/home/nautec/Downloads/TURBID/Photo3D/', label_size=(output_size[0], output_size[1]))

x = tf.placeholder("float", shape=[None, np.prod(np.array(input_size))], name="input_image")
y_ = tf.placeholder("float", shape=[None, np.prod(np.array(output_size))], name="output_image")

sess = tf.InteractiveSession()

deep_dive = DeepDive()
path = '../Local_aux/weights/'

h_conv3 = create_structure(tf, x)

batch_size = 25
learning_rate = 1e-5

y_image = tf.reshape(y_, [-1, output_size[0], output_size[1], output_size[2]])

loss_function = tf.reduce_mean(tf.pow(tf.sub(h_conv3, y_image),2))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

tf.image_summary('inputs', tf.reshape(x, [batch_size, input_size[0], input_size[1], input_size[2]]))
tf.image_summary('outputs(h_conv3)', tf.reshape(h_conv3, [batch_size, output_size[0], output_size[1], output_size[2]]))
tf.scalar_summary('loss', loss_function)


summary_op = tf.merge_all_summaries()

saver = tf.train.Saver(tf.all_variables())

# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

sess.run(tf.initialize_all_variables())

summary_writer = tf.train.SummaryWriter(options.summary_path,
                                            graph_def=sess.graph_def)


for i in range(20000):
  batch = dataset.train.next_batch(batch_size)
  if i%50 == 0:
  	saver.save(sess, 'models/model.ckpt', global_step=i)

  train_accuracy = loss_function.eval(feed_dict={
      x:batch[0], y_: batch[1]})
  
  if i%10 == 0:
  	print("step %d, images used %d, training accuracy %g"%(i, i*batch_size, train_accuracy))
  
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