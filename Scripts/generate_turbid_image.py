import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

"""Visualization libs"""
import matplotlib.cm as cm

import sys
import os
import cv2
import time
import StringIO
from threading import Lock

sys.path.insert(0, os.path.join('/home/nautec/deep-visualization-toolbox'))
from misc import WithTimer
from core import CodependentThread
from image_misc import norm01, norm01c, norm0255, tile_images_normalize, ensure_float01, tile_images_make_tiles, ensure_uint255_and_resize_to_fit, caffe_load_image, get_tiles_height_width
from image_misc import FormattedString, cv2_typeset_text, to_255

sys.path.insert(0, os.path.join('/home/nautec/caffe', 'python'))
import caffe

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# from tensorflow.python.framework import types
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_nn_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *

"""Python libs"""
from optparse import OptionParser


# # Aliases for some automatically-generated names.
# local_response_normalization = gen_nn_ops.lrn


# def deconv2d(value, filter, output_shape, strides, padding="SAME", name=None):
#   """The transpose of `conv2d`.
#   This used to be called "deconvolution", but it is actually the transpose
#   (gradient) of `conv2d`, not an actual deconvolution.
#   Args:
#     value: A 4-D `Tensor` of type `float` and shape
#       `[batch, height, width, in_channels]`.
#     filter: A 4-D `Tensor` with the same type as `value` and shape
#       `[height, width, output_channels, in_channels]`.  `filter`'s
#       `in_channels` dimension must match that of `value`.
#     output_shape: A 1-D `Tensor` representing the output shape of the
#       deconvolution op.
#     strides: A list of ints. The stride of the sliding window for each
#       dimension of the input tensor.
#     padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
#     name: Optional name for the returned tensor.
#   Returns:
#     A `Tensor` with the same type as `value`.
#   Raises:
#     ValueError: If input/output depth does not match `filter`'s shape, or if
#       padding is other than `'VALID'` or `'SAME'`.
#   """
#   with ops.op_scope([value, filter, output_shape], name, "DeConv2D") as name:
#     value = ops.convert_to_tensor(value, name="value")
#     filter = ops.convert_to_tensor(filter, name="filter")
#     if not value.get_shape()[3].is_compatible_with(filter.get_shape()[3]):
#       raise ValueError(
#           "input channels does not match filter's input channels, "
#           "{} != {}".format(value.get_shape()[3], filter.get_shape()[3]))

#     output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
#     if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(4)):
#       raise ValueError("output_shape must have shape (4,), got {}"
#                        .format(output_shape_.get_shape()))

#     if isinstance(output_shape, (list, np.ndarray)):
#       # output_shape's shape should be == [4] if reached this point.
#       if not filter.get_shape()[2].is_compatible_with(output_shape[3]):
#         raise ValueError(
#             "output_shape does not match filter's output channels, "
#             "{} != {}".format(output_shape[3], filter.get_shape()[2]))

#     if padding != "VALID" and padding != "SAME":
#       raise ValueError("padding must be either VALID or SAME:"
#                        " {}".format(padding))

#     return gen_nn_ops.conv2d_backprop_input(input_sizes=output_shape_,
#                                             filter=filter,
#                                             out_backprop=value,
#                                             strides=strides,
#                                             padding=padding,
#                                             name=name)

parser = OptionParser()
parser.add_option("-w", "--weight", dest="weight", default="0",
                  help="weight number in the layer")
(options, args) = parser.parse_args()

def read_eval_data(path):
  """Extract the image and return a 3D uint8 numpy array [y, x, depth]."""
  # print 'Loading image...'
  ims = []
  im = Image.open(path).convert('RGB')
  
  return np.array(im, dtype=np.float32)

path = 'shark.jpg'

x = read_eval_data(path)

net = caffe.Classifier(
            '/home/nautec/deep-visualization-toolbox/models/caffenet-yos/caffenet-yos-deploy.prototxt',
            '/home/nautec/deep-visualization-toolbox/models/caffenet-yos/caffenet-yos-weights', 
            
            #image_dims = (227,227),
        )

sess = tf.InteractiveSession()
weight = int(options.weight)
weight = net.params['conv1'][0].data[weight]

# weight = np.array([list(weight[2]), list(weight[1]), list(weight[0])])
# weight = np.array(weight[0])
weight = np.dstack((weight[2],weight[1],weight[0]))
print weight.shape

# raise
# w_aux = weight.transpose(1,2,0)
# weight = w_aux
# weight = np.fliplr(weight)
# weight = np.flipud(weight)
# print w_aux.shape

plt.imshow(weight)

plt.show()
# print weight

conv_w = tf.constant(weight, shape=[weight.shape[0], weight.shape[1], weight.shape[2], 1])
# conv_w = tf.constant(weight, shape=[weight.shape[1], weight.shape[2], weight.shape[0], 1])
conv_w = tf.Variable(conv_w)

x_image = tf.reshape(x, [-1,256,256,3], "unflattening_reshape")

result = tf.nn.relu(tf.nn.conv2d(x_image, conv_w, strides=[1,1,1,1], padding='VALID'))

sess.run(tf.initialize_all_variables())
sess.run(result)

print result.eval()[0,:,:,0].shape

implot = plt.imshow(result.eval()[0,:,:,0], cmap= cm.gray)
plt.show()

# x = result.eval()
# x_image2 = tf.reshape(x, [1, 256, 256, 1], "name")
# # w = tf.constant(weight, shape=[1, 3, 11, 11])
# conv2_w = tf.constant(weight, shape=[11, 11, 3, 1])
# conv2_w = tf.Variable(conv2_w)

# result = deconv2d(x_image2, conv2_w, (1, 256, 256, 3), strides=[1,1,1,1], padding="SAME", name=None)

# # conv2_w = tf.batch_matrix_inverse(tf.Variable(w))
# # conv2_w = tf.reshape(conv2_w, [11, 11, 1, 3])

# # result = tf.nn.conv2d(x_image2, conv2_w, strides=[1,1,1,1], padding='SAME')

# sess.run(tf.initialize_all_variables())
# sess.run(result)

# print result.eval()[0].shape
# implot = plt.imshow(result.eval()[0])
# plt.show()

