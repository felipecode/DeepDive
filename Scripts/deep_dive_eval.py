import tensorflow as tf
import matplotlib.pyplot as plt
from dirt_or_rain_structure import create_structure
import numpy as np
from PIL import Image

"""Visualization libs"""
import matplotlib.pyplo00t as plt
import matplotlib.cm as cm

def read_eval_data(image_number, i, j):
  """Extract the image and return a 3D uint8 numpy array [y, x, depth]."""
  # print 'Loading image...'
  ims = []
  im = Image.open('/home/nautec/Downloads/TURBID/Photo3D/Training/i' + str(image_number) + 'x' + str(i) + 'y' + str(j) + '.png').convert('RGB')
  
  return np.array(im, dtype=np.float32)

sess = tf.InteractiveSession()

input_size = (64, 64, 3)

x = read_eval_data(3, 20, 40)
h_conv3 = create_structure(tf, x)

cpkt_model = 'models/model.ckpt-750'		# path to the model to be loaded

saver = tf.train.Saver()
saver.restore(sess, cpkt_model)

sess.run(tf.initialize_all_variables())



responses = []
for i in range(1, 40):
  for j in range(1, 58):
    # x = tf.placeholder("float", shape=[None, np.prod(np.array(input_size))], name="input_image")
    x = read_eval_data(3, i, j)

    sess.run(tf.initialize_all_variables())
    sess.run(h_conv3)

    # print h_conv3.eval()[0].shape
    # implot = plt.imshow(h_conv3.eval()[0])
    responses.append(h_conv3.eval()[0])
    # plt.show()

  # print h_conv3.eval().shape
  # implot = plt.imshow(h_conv3.eval()[0])

  # plt.show()

im = np.zeros((1638, 2394, 3))
cont = 0
print len(responses)
print responses[0].shape
for i in range(1, 40):
  for j in range(1, 58):
    im[(i-1)*42:i*42,(j-1)*42:j*42,:] = responses[cont]
    cont += 1
    # implot = plt.imshow(responses[i])
    # plt.show()

implot = plt.imshow(im)
plt.show()

# print h_conv3.eval().shape