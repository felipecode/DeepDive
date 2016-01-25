"""
Implements the deep dive training class
"""

class DeepDiveTrainer(object):

	"""
	save_tensors: list of tensors to save
	"""
	def __init__(self, save_tensors=[]):
		#Loss function
		self.loss = -tf.reduce_sum(y_*tf.log(y_conv))
		
		#Training function
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		#Creates savers
		if save_tensors != []:
			self.saver = tf.train.Saver(save_tensors)

		#Saving:
		# saver.save(sess, 'my-model', global_step=step)

	def train(self):
