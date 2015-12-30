'''
Script to transform a list of files into a .tfrecords
the image_path_file has each line with the path to a file and the label:
path/to/file label
Cheers
'''

from __future__ import print_function
import os
import tensorflow.python.platform
import numpy
import tensorflow as tf
from scipy import misc

tf.app.flags.DEFINE_integer('validation_size', 1,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images, labels, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (dat.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]
    filename = os.path.join('/home/ballester/Documents/DeepDive/Photo3D', name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        label = labels[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _bytes_feature(label),
            # 'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

def main(argv):

    
    #File with the filenames and labels
    images = []
    labels = []
    # image_path_file = 'image_path_file.txt'

    #real number + 1
    max_x = 14
    max_y = 20
    max_im = 21

    path = '/home/ballester/Documents/DeepDive/Photo3D/'

    for i in range(2, max_im):
        for j in range(1, max_y):
            for k in range(1, max_x):
                #open train
                im = misc.imread(path + 'Training/i' + str(i) + 'x' + str(k) + 'y' + str(j) + '.png')
                images.append(im)

                #open gt
                label = misc.imread(path + 'GroundTruth/i1' + 'x' + str(k) + 'y' + str(j) + '.png')
                labels.append(label)



    # Extract it into numpy arrays.  
    train_images = numpy.array(images)
    train_labels = numpy.array(labels)
    # test_images = input_data.extract_images(test_images_filename)
    # test_labels = input_data.extract_labels(test_labels_filename)

    # Generate a validation set.
    # validation_images = train_images[:FLAGS.validation_size, :, :, :]
    # validation_labels = train_labels[:FLAGS.validation_size]
    # train_images = train_images[FLAGS.validation_size:, :, :, :]
    # train_labels = train_labels[FLAGS.validation_size:]
    # Convert to Examples and write the result to TFRecords.
    convert_to(train_images, train_labels, 'train')
    # convert_to(validation_images, validation_labels, 'validation')
    # convert_to(test_images, test_labels, 'test')

if __name__ == '__main__':
  tf.app.run()