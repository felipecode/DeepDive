import input_data_dive

import csv
import tensorflow as tf

import matplotlib.pyplot as plt

sess = tf.InteractiveSession()


#x = tf.placeholder("float", shape=[None, 101568])
#y_ = tf.placeholder("float", shape=[None, 101568])

global_step = tf.Variable(0, trainable=False)
mnist = input_data_dive.read_data_sets()

def read_float_csv(file):
    with open(file, 'rU') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]


def weight_constant(shape,file):

  initial = list(read_float_csv(file))

  weights = tf.constant(initial, shape=shape)

  weights = tf.Variable(weights)
  #initial = tf.cast(initial,tf.float32)
  return weights


def bias_constant(shape,file):

  your_list = list(read_float_csv(file))
  initial = tf.constant(your_list, shape=shape)

  initial = tf.Variable(initial)
  #initial = tf.cast(initial,tf.float32)
  return initial

def bias_variable(shape):  
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='VALID')

def conv2dS(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')




W_conv1 = weight_constant([1,121,1,38],'w_nonoiseC1.csv')



W_conv2 = weight_constant([121,1,38,38],'w_nonoiseC2.csv')





b_conv1 = bias_constant([38],'w_nonoisebc1.csv')


b_conv2 = bias_constant([38],'w_nonoisebc2.csv')

#x_image = tf.reshape(x, [-1,184,184,3])

x = mnist.train._images[0]

#print x.eval()
#print W_conv1.

x = x.reshape(184,184,3)

print x.shape

xR = x[:,:,0].reshape(184*184)
xG = x[:,:,1].reshape(184*184)
xB = x[:,:,2].reshape(184*184)


#xR = x[1:end/3]
#xG = x[(end/3):2*(end/3)]
#xB = x[1:end/3]



x_imageR =  tf.reshape(xR, [-1,184,184,1])

h_convR1 = tf.nn.relu(conv2d(x_imageR, W_conv1) + b_conv1)


h_convR2 = tf.nn.relu(conv2d(h_convR1, W_conv2) + b_conv2)

x_imageG =  tf.reshape(xG, [-1,184,184,1])

h_convG1 = tf.nn.relu(conv2d(x_imageG, W_conv1) + b_conv1)

h_convG2 = tf.nn.relu(conv2d(h_convG1, W_conv2) + b_conv2)

x_imageB =  tf.reshape(xB, [-1,184,184,1])

h_convB1 = tf.nn.relu(conv2d(x_imageB, W_conv1) + b_conv1)

h_convB2 = tf.nn.relu(conv2d(h_convB1, W_conv2) + b_conv2)



h_conv1 = tf.concat(3,[h_convR2,h_convG2,h_convB2]) 


# DENOISE STAGE 




W_noise1 = weight_constant([16,16,114,512],'w_nonoiseC1_sec.csv')


W_noise2 = weight_constant([1,1,512,512],'w_nonoiseC2_sec.csv')


W_noise3 = weight_constant([8,8,512,3],'w_nonoiseC3_sec.csv')




b_noise1 = bias_constant([512],'w_nonoisebc1_sec.csv')


b_noise2 = bias_constant([512],'w_nonoisebc2_sec.csv')


#b_noise3 = bias_constant([192],'w_nonoisebc3_sec.csv')




h_noise1 = tf.nn.relu(conv2d(h_conv1, W_noise1) + b_noise1)



h_noise2 = tf.nn.relu(conv2d(h_noise1, W_noise2) + b_noise2)

print h_noise2


h_noise3 = tf.nn.relu(conv2dS(h_noise2, W_noise3))


#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
sess.run(h_noise3)

print h_noise3.eval().shape

implot = plt.imshow(h_noise3.eval()[0])

plt.show()

#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


#im_summary = tf.image_summary('mnist_images', tf.reshape(h_conv1, [64, 184, 38, 1]))


# h_pool1 = max_pool_2x2(h_conv1)

# W_conv2 = weight_variable([121, 38, 3, 1])
# b_conv2 = bias_variable([38])

# h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

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
