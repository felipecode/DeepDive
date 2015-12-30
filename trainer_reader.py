import tensorflow.python.platform
import tensorflow as tf
import re

from PIL import Image

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100
NUM_EPOCHS_PER_DECAY = 10
batch_size = 50
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.


def read_from_file(tf_records_file):
    class AuxRecord(object):
        pass
    result = AuxRecord()

    # print filenames
    filename_queue = tf.train.string_input_producer([tf_records_file])
    # print dir(filename_queue)

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      dense_keys=['image_raw', 'label'],
      # Defaults are not specified since both keys are required.
      dense_types=[tf.string, tf.string])

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([184*184*3])
    image = tf.cast(image, tf.float32)  

    label = tf.decode_raw(features['label'], tf.uint8)
    label.set_shape([184*184*3])
    label = tf.cast(label, tf.float32)

    # OPTIONAL: Could reshape into a NxN image and apply distortions
    # image = tf.reshape(image, [184, 184, 3])

    min_queue_examples = 2
    print ('Filling queue with %d images' % min_queue_examples)

    return _generate_image_and_label_batch(image, label, min_queue_examples)

def _generate_image_and_label_batch(image, label, min_queue_examples):
  """Construct a queued batch of images and labels.
  Args:
  image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
  label: 1-D Tensor of type.int32
  min_queue_examples: int32, minimum number of samples to retain
    in the queue that provides of batches of examples.
  Returns:
  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
  labels: Labels. 1D tensor of [batch_size] size.
  """


  # creates queue and then
  # read 'FLAGS.batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  images, label_batch = tf.train.batch(
    [image, label],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)
  return images, tf.reshape(label_batch, [batch_size])

def train(total_loss, global_step, learning_rate, lr_decay):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
    Returns:
    train_op: op for training.
    """
    # Variables that affect learning rate.

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(learning_rate,
                                  global_step,
                                  decay_steps,
                                  lr_decay,
                                  staircase=True)
    tf.scalar_summary('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    
    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    return train_op

def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op

# def loss(self, logits, labels):
#     """Add L2Loss to all the trainable variables.
#     Add summary for for "Loss" and "Loss/avg".
#     Args:
#     logits: Logits from inference().
#     labels: Labels from distorted_inputs or inputs(). 1-D tensor
#             of shape [batch_size]
#     Returns:
#     Loss tensor of type float.
#     """
#     # Reshape the labels into a dense Tensor of
#     # shape [batch_size, NUM_CLASSES].
#     sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
#     indices = tf.reshape(tf.range(0, FLAGS.batch_size, 1), [FLAGS.batch_size, 1])
#     concated = tf.concat(1, [indices, sparse_labels])
#     dense_labels = tf.sparse_to_dense(concated,
#                                     [FLAGS.batch_size, NUM_CLASSES],
#                                     1.0, 0.0)
#     # Calculate the average cross entropy loss across the batch.
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#       logits, dense_labels, name='cross_entropy_per_example')
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#     tf.add_to_collection('losses', cross_entropy_mean)
#     # The total loss is defined as the cross entropy loss plus all of the weight
#     # decay terms (L2 loss).
#     return tf.add_n(tf.get_collection('losses'), name='total_loss')