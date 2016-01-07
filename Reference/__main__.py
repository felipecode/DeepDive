"""Deep dive libs"""
from deep_dive import DeepDive
import input_data_dive

"""Core libs"""
import tensorflow as tf
import numpy as np

"""Visualization libs"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm

global_step = tf.Variable(0, trainable=False)
dataset = input_data_dive.read_data_sets()

x = tf.placeholder("float", shape=[184, 184])
y_ = tf.placeholder("float", shape=[184, 184, 3])

sess = tf.InteractiveSession()

deep_dive = DeepDive()
path = '../Local_aux/weights/'

W_conv1 = deep_dive.weight_constant([1,121,1,38], path + 'w_nonoiseC1.csv')

W_smooth = deep_dive.weight_constant([1, 1, 38, 1], path + 'w_nonoiseW_smooth.csv')

W_conv2 = deep_dive.weight_constant([121,1,1,38], path + 'w_nonoiseC2.csv')

b_conv1 = deep_dive.bias_constant([38], path + 'w_nonoisebc1.csv')

b_conv2 = deep_dive.bias_constant([38], path + 'w_nonoisebc2.csv')

#x_image = tf.reshape(x, [-1,184,184,3])

x = dataset.train._images[0]

x = x.reshape(184,184,3)


xR = x[:,:,0].reshape(184*184)
xG = x[:,:,1].reshape(184*184)
xB = x[:,:,2].reshape(184*184)

"""Red Channel"""
x_imageR =  tf.reshape(xR, [-1,184,184,1])
h_convR1 = tf.sigmoid(deep_dive.conv2d(x_imageR, W_conv1) + b_conv1)

#inserting smooth w
h_convR1_concat = tf.reshape(h_convR1[:,:,:,0], [-1, 184, 64, 1])
h_convR1_concat = deep_dive.conv2d(h_convR1_concat, W_conv2)
for i in range(1, 38):
  feature_map = tf.reshape(h_convR1[:,:,:,i], [-1, 184, 64, 1])
  h_convR1_concat = tf.concat(2, [h_convR1_concat, deep_dive.conv2d(feature_map, W_conv2)])

h_convR2 = deep_dive.conv2d(h_convR1_concat, W_smooth)
h_convR2 = tf.sigmoid(tf.reshape(h_convR2, [-1, 64, 64, 38]) + b_conv2)

"""Green Channel"""
x_imageG =  tf.reshape(xG, [-1,184,184,1])
h_convG1 = tf.sigmoid(deep_dive.conv2d(x_imageG, W_conv1) + b_conv1)

#inserting smooth w
h_convG1_concat = tf.reshape(h_convG1[:,:,:,0], [-1, 184, 64, 1])
h_convG1_concat = deep_dive.conv2d(h_convG1_concat, W_conv2)
for i in range(1, 38):
  feature_map = tf.reshape(h_convG1[:,:,:,i], [-1, 184, 64, 1])
  h_convG1_concat = tf.concat(2, [h_convG1_concat, deep_dive.conv2d(feature_map, W_conv2)])

h_convG2 = deep_dive.conv2d(h_convG1_concat, W_smooth)
h_convG2 = tf.sigmoid(tf.reshape(h_convG2, [-1, 64, 64, 38]) + b_conv2)


"""Blue Channel"""
x_imageB =  tf.reshape(xB, [-1,184,184,1])
h_convB1 = tf.sigmoid(deep_dive.conv2d(x_imageB, W_conv1) + b_conv1)

#inserting smooth w
h_convB1_concat = tf.reshape(h_convB1[:,:,:,0], [-1, 184, 64, 1])
h_convB1_concat = deep_dive.conv2d(h_convB1_concat, W_conv2)
for i in range(1, 38):
  feature_map = tf.reshape(h_convB1[:,:,:,i], [-1, 184, 64, 1])
  h_convB1_concat = tf.concat(2, [h_convB1_concat, deep_dive.conv2d(feature_map, W_conv2)])

h_convB2 = deep_dive.conv2d(h_convB1_concat, W_smooth)
h_convB2 = tf.sigmoid(tf.reshape(h_convB2, [-1, 64, 64, 38]) + b_conv2)


#Concatenating color channels
h_conv1 = tf.concat(3,[h_convR2,h_convG2,h_convB2]) 


# DENOISE STAGE 

# sess.run(tf.initialize_all_variables())
# sess.run(h_convR2)

# print h_convR2.eval().shape

# implot = plt.imshow(h_convR2.eval()[0,:,:,14],cmap= cm.Greys_r)

# plt.show()


"""Loading weights"""
W_noise1 = deep_dive.weight_constant([16,16,114,512], path + 'w_nonoiseC1_sec.csv')
W_noise2 = deep_dive.weight_constant([1,1,512,512], path + 'w_nonoiseC2_sec.csv')
W_noise3 = deep_dive.weight_constant([8,8,512,192], path + 'w_nonoiseC3_sec.csv')

"""Loading biases"""
b_noise1 = deep_dive.bias_constant([512], path + 'w_nonoisebc1_sec.csv')
b_noise2 = deep_dive.bias_constant([512], path + 'w_nonoisebc2_sec.csv')
b_noise3 = deep_dive.bias_constant([192], path + 'w_nonoisebc3_sec.csv')

"""Convolutions"""
h_noise1 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_noise1) + b_noise1)
h_noise2 = tf.sigmoid(deep_dive.conv2d(h_noise1, W_noise2) + b_noise2)
h_noise3 = tf.sigmoid(deep_dive.conv2d(h_noise2, W_noise3, padding="SAME") + b_noise3)

#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
sess.run(h_noise3)

# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

loss_function = tf.sub(h_noise3, y_)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


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