import numpy as np
import matplotlib.pyplot as plt
import time
from config import *
from random import *
import colorsys
import os
import json

def generate_ncolors(num_colors):

	color_pallet = []
	for i  in range(0,360, 360 / num_colors):
		hue = i
		saturation = 90 + float(randint(0,1000))/1000 * 10
		lightness = 50 + float(randint(0,1000))/1000 * 10

		color = colorsys.hsv_to_rgb(float(hue)/360.0,saturation/100,lightness/100) 

		color_pallet.append(color)

	return color_pallet

""" Here I read the files """
config = configMain()

if os.path.isfile(config.models_path +'summary.json'):
      outfile= open(config.models_path +'summary.json','r')
      dados=json.load(outfile)
else:
	print 'Arquivo inexistente.'

variable_errors=dados['variable_errors']
variable_errors_val=dados['variable_errors_val']
time=dados['time']
summary_writing_period=dados['summary_writing_period']
validation_period=dados['validation_period']
batch_size=dados['batch_size']

 
#train
batch_number = range(0,len(variable_errors)*summary_writing_period,summary_writing_period)
plt.figure(1)
plt.subplot(211)
plt.plot(batch_number, variable_errors, 'b')
axes = plt.gca()
axes.set_ylim([0,1])
plt.title('Train')
plt.grid(True)

#validation
batch_number_val = range(validation_period,(len(variable_errors_val)+1)*validation_period,validation_period)
plt.subplot(212)
plt.plot(batch_number_val, variable_errors_val, 'r')
axes = plt.gca()
axes.set_ylim([0,1])
plt.title('Validation')
plt.grid(True)

#validation
# x = range(10)
# error_per_transmission = dados['error_per_transmission']
# plt.subplot(313)
# plt.bar(x, error_per_transmission, color='blue')
# axes = plt.gca()
# axes.set_xlim([0,config.num_bins])
# plt.title('Error per transmission mean')
# plt.grid(True)

plt.show()
