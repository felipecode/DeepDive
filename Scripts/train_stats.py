import numpy as np
import matplotlib.pyplot as plt
import time
from config import *
from random import *
import colorsys

config = configMain()

def generate_ncolors(num_colors):

	color_pallet = []
	for i  in range(0,360, 360 / num_colors):

		hue = i
		saturation = 90 + float(randint(0,1000))/1000 * 10
		lightness = 50 + float(randint(0,1000))/1000 * 10

		color = colorsys.hsv_to_rgb(float(hue)/360.0,saturation/100,lightness/100) 


		color_pallet.append(color)

		#addColor(c);
	return color_pallet




def plot_neural_stats(batch_number,time,variable_errors,variable_val_errors,end_epoch_positions,avg_error_epoch,avg_error_val_epoch):

	x1 = batch_number
	x2 = time
	xVal = range(1,len(variable_errors[0])*config.summary_writing_period,config.validation_period)


	xAvgError = range(0,len(avg_error_epoch)*config.summary_writing_period*(config.dataset_train_size/config.batch_size),config.summary_writing_period*(config.dataset_train_size/config.batch_size))
	#xAvgError_val = range(0,len(avg_error_epoch)*config.summary_writing_period*(5800/config.batch_size),config.summary_writing_period*(5800/config.batch_size))
	
	#yplot1_val = validation_error
	#yplot1_train = training_error  # Energy

	lines = end_epoch_positions
	
	subtitle = config.variable_names

	#axes.get_ylim() 


	""" Variable errors should be a vector of error vectors """
	fig = plt.figure()
	ax1_1 = fig.add_subplot(2, 1, 1)
	ax1_2 = ax1_1.twiny()
	variable_val_errors[0].append(0)
	#print len(variable_errors)
	color_pallet = generate_ncolors(len(variable_errors))

	for i in range(0,len(variable_errors)):
		#print x1
		print len(variable_errors[i])
		print len(variable_val_errors[i])
		print len(x1)
		print len(xVal)
		print subtitle[i]
		ax1_1.plot(x1,variable_errors[i],'-',color=color_pallet[i],label=subtitle[i])
		ax1_1.plot(xVal,variable_val_errors[i],'--',color=color_pallet[i],label=subtitle[i]+' Validation')

	ax2_1 = fig.add_subplot(2, 1, 2)


	print len(avg_error_epoch)
	print len(avg_error_val_epoch)
	ax2_1.plot(xAvgError,avg_error_epoch,'r-',xAvgError,avg_error_val_epoch,'r--')



	for i in range(0,len(end_epoch_positions)):
		ax1_1.plot([end_epoch_positions[i],end_epoch_positions[i]],[0,1],'k-')


	legend = ax1_1.legend(loc='upper center', shadow=True)
	legend.get_frame().set_linewidth(3.0)
	#legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
	#ax1_2.set_xlim(ax1_1.get_xlim())
	ax1_2.set_ylim([0,1])
	ax1_2.set_xticklabels(x2)
	ax1_2.set_xlabel('Time (min)')
	ax1_1.set_xlabel('Batch Number')
	ax2_1.set_ylabel('Average Error Per Epoch')
	ax2_1.set_xlabel('Batch Number'),
	ax2_1.set_ylim([0,1])

	plt.show()


""" Here I read the files """

with open(config.models_path + "variable_errors") as f:
    variable_errors =[map(float, f)]

with open(config.models_path + "variable_errors_val") as f:
    variable_errors_val = [map(float, f)]

with open(config.models_path + "time") as f:
    time = map(float, f)



batchs_per_epoch = config.dataset_train_size/config.batch_size
print batchs_per_epoch

batch_number = range(1,len(variable_errors[0])*config.summary_writing_period,config.summary_writing_period)
end_epoch_positions = range(1,len(variable_errors[0])*config.summary_writing_period,batchs_per_epoch)
avg_error_epoch = []
avg_error_val_epoch = []


for i in range(0,len(end_epoch_positions)):
	
	initial_train =i*(batchs_per_epoch/config.summary_writing_period)
	final_train =(i+1)*(batchs_per_epoch/config.summary_writing_period)
	initial_val = i*(batchs_per_epoch/config.validation_period)
	final_val = (i+1)*(batchs_per_epoch/config.validation_period)

	avg_point = sum(variable_errors[0][initial_train:final_train])/(batchs_per_epoch/config.summary_writing_period)
	avg_val_point = sum(variable_errors_val[0][initial_val:final_val])/(batchs_per_epoch/config.validation_period)
	avg_error_epoch.append(avg_point)
	avg_error_val_epoch.append(avg_val_point)



#print avg_error_epoch
#batch_number = [0,1,2,3,4,5]
#time = [10,20,30,40,50,60]
#variable_errors = [[1,0.8,0.6,0.2,0.1,0.095],[0.7,0.2,0.1,0.05,0.01,0.005]]
#end_epoch_positions = [3]

plot_neural_stats(batch_number,time,variable_errors,variable_errors_val,end_epoch_positions,avg_error_epoch,avg_error_val_epoch)