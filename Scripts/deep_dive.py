import csv
import tensorflow as tf
import numpy as np

#x = tf.placeholder("float", shape=[None, 101568])
#y_ = tf.placeholder("float", shape=[None, 101568])

class DeepDive(object):

  def __init__(self):
    pass

  """Reads floats from a csv file (auxiliar function)"""
  def read_float_csv(self, file):
    with open(file, 'rU') as data:
      reader = csv.reader(data)
      for row in reader:
        yield [ float(i) for i in row ]

  """
  Loads weights from a csv file
  shape: 4D list as [kernel_size_x, kernel_size_y, ]
  """
  def weight_constant(self, shape,file):
    initial = list(self.read_float_csv(file))
    weights = tf.constant(initial, shape=shape)

    weights = tf.Variable(weights)
    return weights

  """Creates a weight variable (TODO)"""
  def weight_variable(self, shape, file):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  """Loads bias from a csv file"""
  def bias_constant(self, shape,file):
    initial = list(self.read_float_csv(file))
    bias = tf.constant(initial, shape=shape)

    bias = tf.Variable(bias)
    return bias

  """Creates a bias variable (TODO)"""
  def bias_variable(self, shape):  
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  """
  Creates a 2d Convolution layer.
  x: input layer
  padding: 'same' or 'valid'
  W: variable or constant weight created.
  """
  def conv2d(self, x, W, strides=[1,1,1,1], padding='VALID'):
    return tf.nn.conv2d(x, W, strides=strides ,padding=padding)

  """
  Creates a 2d Depth Wise Convolution layer.
  """
  def dWiseConv2d(self, x, W, strides=[1,1,1,1], padding='VALID'):
    return tf.nn.depthwise_conv2d(x, W, strides=strides, padding=padding)

  """
  Creates a max pooling layer
  x: input layer
  """
  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')




#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


#im_summary = tf.image_summary('mnist_images', tf.reshape(h_conv1, [64, 184, 38, 1]))


# h_pool1 = max_pool_2x2(h_conv1)

# W_conv2 = weight_variable([121, 38, 3, 1])
# b_conv2 = bias_variable([38])

# h_conv2 = tf.sigmoid(conv2d(h_conv1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])

# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# loss = -tf.reduce_sum(y_*tf.log(y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#summary
# saver = tf.train.Saver(tf.all_variables())
# c_summary = tf.scalar_summary('loss', loss)
# im_summary = tf.image_summary('mnist_images', tf.reshape(mnist.train.images, [55000, 28, 28, 1]))

# merged = tf.merge_all_summaries()
# writer = tf.train.SummaryWriter("/tmp/deep_dive_logs/", sess.graph_def)

# #train_op = tr.train(loss, global_step, learning_rate=0.5, lr_decay=0.01)

# sess.run(tf.initialize_all_variables())
# tf.train.start_queue_runners(sess=sess)

# for i in range(1):
#   if i%100 == 0:
#     # train_accuracy = accuracy.eval(feed_dict={
#     #     x:batch[0], y_: batch[1], keep_prob: 1.0})
#     # print("step %d, training accuracy %g"%(i, train_accuracy))
#     result = sess.run([merged])
#     writer.add_summary(result[0], i)  

# sess.close()

# for i in range(20000):
#   # batch = mnist.train.next_batch(50)
#   # feed = {x: batch[0], y_: batch[1], keep_prob: 0.5}

#   if i%100 == 0:
#     # train_accuracy = accuracy.eval(feed_dict={
#     #     x:batch[0], y_: batch[1], keep_prob: 1.0})
#     # print("step %d, training accuracy %g"%(i, train_accuracy))
#     result = sess.run([merged, accuracy])
#     writer.add_summary(result[0], i)


#   else:
#     # sess.run(train_step, feed_dict=feed)
#     _, loss_value = sess.run([train_op, loss])

#   if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
#     checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
#     saver.save(sess, checkpoint_path, global_step=step)

# print("test accuracy %g"%accuracy.eval(feed_dict={
    # x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
