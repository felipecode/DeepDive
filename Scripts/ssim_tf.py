

""" This implementation is single channel """

def ssim_tf(tf,I1,I2):


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

	ssim = tf.reduce_sum(ssim_map)
	print tf.shape(ssim_map)
	ssim = tf.div(ssim, 128*128*3)
	ssim = tf.sub(1.0,ssim)
	return ssim

# Scalar getMSSIM_GPU_optimized(const Mat& i1, const Mat& i2, BufferMSSIM& b)
# {
# 	const float C1 = 6.5025f, C2 = 58.5225f;
# 	/***************************** INITS **********************************/

# 	b.gI1.upload(i1);
# 	b.gI2.upload(i2);

# 	gpu::Stream stream;

# 	stream.enqueueConvert(b.gI1, b.t1, CV_32F);
# 	stream.enqueueConvert(b.gI2, b.t2, CV_32F);

# 	gpu::split(b.t1, b.vI1, stream);
# 	gpu::split(b.t2, b.vI2, stream);
# 	Scalar mssim;

# 	gpu::GpuMat buf;

# 	for (int i = 0; i < b.gI1.channels(); ++i)
# 	{
# 		gpu::multiply(b.vI2[i], b.vI2[i], b.I2_2, stream);        // I2^2
# 		gpu::multiply(b.vI1[i], b.vI1[i], b.I1_2, stream);        // I1^2
# 		gpu::multiply(b.vI1[i], b.vI2[i], b.I1_I2, stream);       // I1 * I2

# 		gpu::GaussianBlur(b.vI1[i], b.mu1, Size(11, 11), buf, 1.5, 0, BORDER_DEFAULT, -1, stream);
# 		gpu::GaussianBlur(b.vI2[i], b.mu2, Size(11, 11), buf, 1.5, 0, BORDER_DEFAULT, -1, stream);

# 		gpu::multiply(b.mu1, b.mu1, b.mu1_2, stream);
# 		gpu::multiply(b.mu2, b.mu2, b.mu2_2, stream);
# 		gpu::multiply(b.mu1, b.mu2, b.mu1_mu2, stream);

# 		gpu::GaussianBlur(b.I1_2, b.sigma1_2, Size(11, 11), buf, 1.5, 0, BORDER_DEFAULT, -1, stream);
# 		gpu::subtract(b.sigma1_2, b.mu1_2, b.sigma1_2, gpu::GpuMat(), -1, stream);
# 		//b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation

# 		gpu::GaussianBlur(b.I2_2, b.sigma2_2, Size(11, 11), buf, 1.5, 0, BORDER_DEFAULT, -1, stream);
# 		gpu::subtract(b.sigma2_2, b.mu2_2, b.sigma2_2, gpu::GpuMat(), -1, stream);
# 		//b.sigma2_2 -= b.mu2_2;

# 		gpu::GaussianBlur(b.I1_I2, b.sigma12, Size(11, 11), buf, 1.5, 0, BORDER_DEFAULT, -1, stream);
# 		gpu::subtract(b.sigma12, b.mu1_mu2, b.sigma12, gpu::GpuMat(), -1, stream);
# 		//b.sigma12 -= b.mu1_mu2;

# 		//here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
# 		gpu::multiply(b.mu1_mu2, 2, b.t1, 1, -1, stream); //b.t1 = 2 * b.mu1_mu2 + C1;
# 		gpu::add(b.t1, C1, b.t1, gpu::GpuMat(), -1, stream);
# 		gpu::multiply(b.sigma12, 2, b.t2, 1, -1, stream); //b.t2 = 2 * b.sigma12 + C2;
# 		gpu::add(b.t2, C2, b.t2, gpu::GpuMat(), -12, stream);

# 		gpu::multiply(b.t1, b.t2, b.t3, 1, -1, stream);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

# 		gpu::add(b.mu1_2, b.mu2_2, b.t1, gpu::GpuMat(), -1, stream);
# 		gpu::add(b.t1, C1, b.t1, gpu::GpuMat(), -1, stream);

# 		gpu::add(b.sigma1_2, b.sigma2_2, b.t2, gpu::GpuMat(), -1, stream);
# 		gpu::add(b.t2, C2, b.t2, gpu::GpuMat(), -1, stream);


# 		gpu::multiply(b.t1, b.t2, b.t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
# 		gpu::divide(b.t3, b.t1, b.ssim_map, 1, -1, stream);      // ssim_map =  t3./t1;

# 		stream.waitForCompletion();

# 		Scalar s = gpu::sum(b.ssim_map, b.buf);
# 		mssim.val[i] = s.val[0] / (b.ssim_map.rows * b.ssim_map.cols);

# 	}
# 	return mssim;
# }