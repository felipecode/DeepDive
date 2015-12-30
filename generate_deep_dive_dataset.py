'''
Script to transform a list of files into a .tfrecords
the image_path_file has each line with the path to a file and the label:
path/to/file label
Cheers
'''

# from __future__ import print_function
import os
from time import sleep
import tensorflow.python.platform
import numpy
import tensorflow as tf
from scipy import misc

tf.app.flags.DEFINE_integer('validation_size', 1,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
FLAGS = tf.app.flags.FLAGS
        

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
        print 'Image: ', i
        for j in range(1, max_y):
            for k in range(1, max_x):
                #open train
                im = misc.imread(path + 'Training/i' + str(i) + 'x' + str(k) + 'y' + str(j) + '.png')
                images.append(im)

                #open gt
                label = misc.imread(path + 'GroundTruth/i1' + 'x' + str(k) + 'y' + str(j) + '.png')
                labels.append(label)


    # import csv
    # with open('images.csv', 'wb') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=' ',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    #     for line in images:
    #         spamwriter.writerow(line)

    # with open('labels.csv', 'wb') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=' ',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for line in labels:
    #         spamwriter.writerow(line)

    # Extract it into numpy arrays.  
    # train_images = numpy.array(images)
    # train_labels = numpy.array(labels)
    # test_images = input_data.extract_images(test_images_filename)
    # test_labels = input_data.extract_labels(test_labels_filename)

    # Generate a validation set.
    # validation_images = train_images[:FLAGS.validation_size, :, :, :]
    # validation_labels = train_labels[:FLAGS.validation_size]
    # train_images = train_images[FLAGS.validation_size:, :, :, :]
    # train_labels = train_labels[FLAGS.validation_size:]
    # Convert to Examples and write the result to TFRecords.
    # convert_to(train_images, train_labels, 'train')
    # convert_to(validation_images, validation_labels, 'validation')
    # convert_to(test_images, test_labels, 'test')

if __name__ == '__main__':
  tf.app.run()