from loss_network import *
def L1_distance(tf, output, ground_truth):
	return tf.reduce_mean(tf.abs(tf.sub(output, ground_truth)), reduction_indices=[1,2,3])

def mse(tf, output, ground_truth):
	return tf.reduce_mean(tf.squared_difference(output, ground_truth), reduction_indices=[1,2,3])

def distance(tf, output, ground_truth):
	return tf.sqrt(tf.reduce_sum(tf.squared_difference(output, ground_truth), reduction_indices=[1,2,3])

def pixel_distance_mean(tf, output, ground_truth):
	return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(output, ground_truth),reduction_indices=[3])), reduction_indices=[1,2])

def feature_loss(tf, output, ground_truth, sess):
	return create_loss_structure(tf, 255.0*last_layer, 255.0*y_, sess)/255.0

def ssim_loss(tf,I1,I2):


	C1 = 6.5025
	C2 = 58.5225

	print I1


	I1_2 = tf.mul(I1, I1) # square the image
	I2_2 = tf.mul(I2, I2)
	I12 = tf.mul(I1, I2)
	# clur the images

	W = [[[0.0000,0.0000, 0.0000 ,   0.0001 ,   0.0002  ,  0.0003  ,  0.0002  ,  0.0001 ,   0.0000 ,   0.0000  ,  0.0000],
    [0.0000 ,   0.0001,    0.0003 ,   0.0008,    0.0016  ,  0.0020,    0.0016   , 0.0008,    0.0003  ,  0.0001,    0.0000],
    [0.0000  ,  0.0003 ,   0.0013 ,   0.0039  ,  0.0077 ,   0.0096 ,   0.0077   , 0.0039,    0.0013  ,  0.0003,    0.0000],
    [0.0001   , 0.0008  ,  0.0039 ,   0.0120  ,  0.0233 ,   0.0291 ,   0.0233   , 0.0120 ,   0.0039  ,  0.0008,    0.0001],
    [0.0002   , 0.0016   , 0.0077 ,   0.0233  ,  0.0454  ,  0.0567 ,   0.0454   , 0.0233  ,  0.0077  ,  0.0016,    0.0002],
    [0.0003   , 0.0020  ,  0.0096 ,   0.0291  ,  0.0567  ,  0.0708 ,   0.0567   , 0.0291  ,  0.0096  ,  0.0020 ,   0.0003],
    [0.0002   , 0.0016   , 0.0077 ,   0.0233  ,  0.0454 ,   0.0567 ,   0.0454   , 0.0233  ,  0.0077  ,  0.0016 ,   0.0002],
    [0.0001   , 0.0008   , 0.0039 ,   0.0120  ,  0.0233 ,   0.0291 ,   0.0233   , 0.0120  ,  0.0039  ,  0.0008 ,   0.0001],
    [0.0000   , 0.0003   , 0.0013 ,   0.0039  ,  0.0077  ,  0.0096 ,   0.0077   , 0.0039  ,  0.0013  ,  0.0003 ,   0.0000],
    [0.0000   , 0.0001   , 0.0003 ,   0.0008  ,  0.0016  ,  0.0020 ,   0.0016   , 0.0008  ,  0.0003  ,  0.0001 ,   0.0000],
    [0.0000   , 0.0000   , 0.0000,    0.0001  ,  0.0002,    0.0003 ,   0.0002 ,   0.0001  ,  0.0000 ,   0.0000 ,   0.0000]]]


	W = tf.concat(0,[W, W, W ])
	W =  tf.reshape(W, [11,11,3,1])


    # create a typical 11x11 gausian kernel with 1.5 sigma
	MU1 = tf.nn.conv2d(I1, W, strides=[1,1,1,1] ,padding='VALID')
	MU2 = tf.nn.conv2d(I2, W, strides=[1,1,1,1] ,padding='VALID')

	MU1_2 = tf.mul(MU1, MU1) # square the image
	MU2_2 = tf.mul(MU2, MU2)
	MU12 = tf.mul(MU1, MU2)

	sigma1_2 = tf.nn.conv2d(I1_2, W, strides=[1,1,1,1] ,padding='VALID')
	sigma1_2 = tf.sub(sigma1_2 , MU1_2)


	sigma2_2 = tf.nn.conv2d(I2_2, W, strides=[1,1,1,1] ,padding='VALID')
	sigma2_2 = tf.sub(sigma2_2 , MU2_2)


	sigma12 = tf.nn.conv2d(I12, W, strides=[1,1,1,1] ,padding='VALID')
	sigma12 = tf.sub(sigma12 , MU12)

	t1  = tf.mul(MU12, 2)
	t1  = tf.add(t1,C1)
	t2  = tf.mul(sigma12,2)
	t2  = tf.add(t2,C2)
	t3 = tf.mul(t1,t2)

	t1 = tf.add(MU1_2,MU2_2)
	t1  = tf.add(t1,C1)
	t2 = tf.add(sigma1_2,sigma2_2)
	t2  = tf.add(t2,C2)

	t1 = tf.mul(t1,t2)

	ssim_map = tf.div(t3,t1)
	ssim = tf.reduce_mean(ssim_map)
	ssim = tf.sub(1.0,ssim)
	return ssim



