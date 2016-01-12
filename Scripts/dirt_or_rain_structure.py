def create_structure(tf, x, y_):
 
  """Deep dive libs"""
  from deep_dive import DeepDive
  import input_data_dive

  # global_step = tf.Variable(0, trainable=False, name="global_step")

  # Our little piece of network for ultimate underwater deconvolution and domination of the sea-world

  deep_dive = DeepDive()
  path = '../Local_aux/weights/'

  W_conv1 = deep_dive.weight_variable([16,16,3,512])

  #W_smooth = deep_dive.weight_variable([1, 1, 38, 1])

  W_conv2 = deep_dive.weight_variable([1,1,512,512])

  W_conv3 = deep_dive.weight_variable([8,8,512,3])

  b_conv1 = deep_dive.bias_variable([512])

  b_conv2 = deep_dive.bias_variable([512])

  b_conv3 = deep_dive.bias_variable([3])

  #x_image = tf.reshape(x, [-1,184,184,3])


  x_image = tf.reshape(x, [-1,64,64,3], "unflattening_reshape")

  """Red Channel"""
  # x_imageR =  tf.reshape(xR, [-1,184,184,1])
  h_conv1 = tf.nn.relu(deep_dive.conv2d(x_image, W_conv1) + b_conv1, name="first_sigmoid")
  h_conv2 = tf.nn.relu(deep_dive.conv2d(h_conv1, W_conv2) + b_conv2, name="second_sigmoid")



  h_conv3 = tf.nn.relu(deep_dive.conv2d(h_conv2, W_conv3, padding='SAME') + b_conv3, name="third_sigmoid")


  return h_conv3

  # batch_size = 50

  # y_image = tf.reshape(y_, [-1,56,56,1])

  # loss_function = tf.reduce_mean(tf.pow(tf.sub(h_conv3, y_image),2))

  # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)

  # tf.image_summary('inputs', tf.reshape(x, [batch_size, 184, 184, 1]))
  # tf.image_summary('outputs(h_conv3)', tf.reshape(h_conv3, [batch_size, 56, 56, 1]))
  # tf.scalar_summary('loss', loss_function)


  # summary_op = tf.merge_all_summaries()

  # saver = tf.train.Saver(tf.all_variables())

  # # keep_prob = tf.placeholder("float")
  # # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # sess.run(tf.initialize_all_variables())

  # summary_writer = tf.train.SummaryWriter('/tmp/deep_dive_1',
  #                                             graph_def=sess.graph_def)

  # for i in range(20000):
  #   batch = dataset.train.next_batch(batch_size)
  #   if i%50 == 0:
  #   	saver.save(sess, 'model.ckpt', global_step=i)

  #   train_accuracy = loss_function.eval(feed_dict={
  #       x:batch[0], y_: batch[1]})
  #   print("step %d, training accuracy %g"%(i, train_accuracy))
  #   train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  #   summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1]})
  #   summary_writer.add_summary(summary_str, i)


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