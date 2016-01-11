import tensorflow as tf

cpkt_model = ''		# path to the model to be loaded

sess = tf.InteractiveSession()
saver.restore(sess, cpkt_model)

sess.run(tf.initialize_all_variables())
sess.run(h_conv3)

print h_conv3.eval().shape

implot = plt.imshow(h_convR2.eval()[0,:,:,14],cmap= cm.Greys_r)

plt.show()