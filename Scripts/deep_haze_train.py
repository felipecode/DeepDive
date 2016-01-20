import tensorflow as tf
from deep_dive import DeepDive
import input_data_haze
from deep_haze_alex import create_structure

sess = tf.InteractiveSession()

global_step = tf.Variable(0, trainable=False, name="global_step")

dataset = input_data_haze.read_data_sets()

x = tf.placeholder("float", shape=[None, 184*184*3], name="input_image")
y_ = tf.placeholder("float", shape=[None, 2], name="output")

y_conv = create_structure(tf, x)

batch_size = 15

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.image_summary('inputs', tf.reshape(x, [batch_size, 184, 184, 3]))
tf.scalar_summary('accuracy', accuracy)
summary_op = tf.merge_all_summaries()

summary_writer = tf.train.SummaryWriter('/tmp/deep_haze_2',
                                            graph_def=sess.graph_def)


saver = tf.train.Saver(tf.all_variables())
sess.run(tf.initialize_all_variables())


for i in range(20000):
  batch = dataset.train.next_batch(batch_size)
  if i%1000 == 0:	
    saver.save(sess, 'models_haze/model.ckpt', global_step=i)
    print('Model saved.')
  
  if i%50 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

  # summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1]})
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
  summary_writer.add_summary(summary_str, i)
