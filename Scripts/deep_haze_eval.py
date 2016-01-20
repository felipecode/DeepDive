import tensorflow as tf
import matplotlib.pyplot as plt
from deep_haze import create_structure
import numpy as np
from PIL import Image

def read_eval_data(path):
  """Extract the image and return a 3D uint8 numpy array [y, x, depth]."""
  # print 'Loading image...'
  ims = []
  im = Image.open(path).convert('RGB')
  im = im.resize((184, 184), Image.ANTIALIAS)

  return np.array(im, dtype=np.float32)

sess = tf.InteractiveSession()

x = read_eval_data('/home/nautec/NoWater/ILSVRC2012_test_00000616.JPEG')
y_conv = create_structure(tf, x)

cpkt_model = 'models_haze/model.ckpt-1000'		# path to the model to be loaded

saver = tf.train.Saver()
saver.restore(sess, cpkt_model)

sess.run(tf.initialize_all_variables())

result = sess.run(y_conv)
print result