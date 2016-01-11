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

x = tf.placeholder("float", shape=[None, 184*184])
y_ = tf.placeholder("float", shape=[None, 56*56])

sess = tf.InteractiveSession()

deep_dive = DeepDive()
path = '../Local_aux/weights/'


# Our little piece of network for ultimate underwater deconvolution and domination of the sea-world

W_conv1 = deep_dive.weight_variable([1,121,1,38])

#W_smooth = deep_dive.weight_variable([1, 1, 38, 1])

W_conv2 = deep_dive.weight_variable([121,1,38,38])

W_conv3 = deep_dive.weight_variable([9,9,38,1])

b_conv1 = deep_dive.bias_variable([38])

b_conv2 = deep_dive.bias_variable([38])

b_conv3 = deep_dive.bias_variable([1])

#x_image = tf.reshape(x, [-1,184,184,3])


x_image = tf.reshape(x, [-1,184,184,1])

"""Red Channel"""
# x_imageR =  tf.reshape(xR, [-1,184,184,1])
h_conv1 = tf.sigmoid(deep_dive.conv2d(x_image, W_conv1) + b_conv1)
h_conv2 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_conv2) + b_conv2)



h_conv3 = tf.sigmoid(deep_dive.conv2d(h_conv2, W_conv3) + b_conv3)


print  h_conv3

#inserting smooth w
# h_convR1_concat = tf.reshape(h_convR1[:,:,:,0], [-1, 184, 64, 1])
# h_convR1_concat = deep_dive.conv2d(h_convR1_concat, W_conv2)
# for i in range(1, 38):
#   feature_map = tf.reshape(h_convR1[:,:,:,i], [-1, 184, 64, 1])
#   h_convR1_concat = tf.concat(2, [h_convR1_concat, deep_dive.conv2d(feature_map, W_conv2)])

# h_convR2 = deep_dive.conv2d(h_convR1_concat, W_smooth)
# h_convR2 = tf.sigmoid(tf.reshape(h_convR2, [-1, 64, 64, 38]) + b_conv2)

# """Green Channel"""
# x_imageG =  tf.reshape(xG, [-1,184,184,1])
# h_convG1 = tf.sigmoid(deep_dive.conv2d(x_imageG, W_conv1) + b_conv1)

# #inserting smooth w
# h_convG1_concat = tf.reshape(h_convG1[:,:,:,0], [-1, 184, 64, 1])
# h_convG1_concat = deep_dive.conv2d(h_convG1_concat, W_conv2)
# for i in range(1, 38):
#   feature_map = tf.reshape(h_convG1[:,:,:,i], [-1, 184, 64, 1])
#   h_convG1_concat = tf.concat(2, [h_convG1_concat, deep_dive.conv2d(feature_map, W_conv2)])

# h_convG2 = deep_dive.conv2d(h_convG1_concat, W_smooth)
# h_convG2 = tf.sigmoid(tf.reshape(h_convG2, [-1, 64, 64, 38]) + b_conv2)


# """Blue Channel"""
# x_imageB =  tf.reshape(xB, [-1,184,184,1])
# h_convB1 = tf.sigmoid(deep_dive.conv2d(x_imageB, W_conv1) + b_conv1)

# #inserting smooth w
# h_convB1_concat = tf.reshape(h_convB1[:,:,:,0], [-1, 184, 64, 1])
# h_convB1_concat = deep_dive.conv2d(h_convB1_concat, W_conv2)
# for i in range(1, 38):
#   feature_map = tf.reshape(h_convB1[:,:,:,i], [-1, 184, 64, 1])
#   h_convB1_concat = tf.concat(2, [h_convB1_concat, deep_dive.conv2d(feature_map, W_conv2)])

# h_convB2 = deep_dive.conv2d(h_convB1_concat, W_smooth)
# h_convB2 = tf.sigmoid(tf.reshape(h_convB2, [-1, 64, 64, 38]) + b_conv2)


# #Concatenating color channels
# h_conv1 = tf.concat(3,[h_convR2,h_convG2,h_convB2]) 


# DENOISE STAGE 

# sess.run(tf.initialize_all_variables())
# sess.run(h_convR2)

# print h_convR2.eval().shape

# implot = plt.imshow(h_convR2.eval()[0,:,:,14],cmap= cm.Greys_r)

# plt.show()


# """Loading weights"""
# W_noise1 = deep_dive.weight_constant([16,16,114,512], path + 'w_nonoiseC1_sec.csv')
# W_noise2 = deep_dive.weight_constant([1,1,512,512], path + 'w_nonoiseC2_sec.csv')
# W_noise3 = deep_dive.weight_constant([8,8,512,192], path + 'w_nonoiseC3_sec.csv')

# """Loading biases"""
# b_noise1 = deep_dive.bias_constant([512], path + 'w_nonoisebc1_sec.csv')
# b_noise2 = deep_dive.bias_constant([512], path + 'w_nonoisebc2_sec.csv')
# b_noise3 = deep_dive.bias_constant([192], path + 'w_nonoisebc3_sec.csv')

# """Convolutions"""
# h_noise1 = tf.sigmoid(deep_dive.conv2d(h_conv1, W_noise1) + b_noise1)
# h_noise2 = tf.sigmoid(deep_dive.conv2d(h_noise1, W_noise2) + b_noise2)
# h_noise3 = tf.sigmoid(deep_dive.conv2d(h_noise2, W_noise3, padding="SAME") + b_noise3)

#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


y_image = tf.reshape(y_, [-1,56,56,1])

loss_function = tf.reduce_mean(tf.pow(tf.sub(h_conv3, y_image),2))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)


# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

sess.run(tf.initialize_all_variables())


for i in range(20000):
  batch = dataset.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = loss_function.eval(feed_dict={
        x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})



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