import numpy as np
import matplotlib.pyplot as plt
import time
import math
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
config = configVisualization()

if os.path.isfile(config.summary_path +'summary.json'):
      outfile= open(config.summary_path +'summary.json','r')
      dados=json.load(outfile)
else:
	print 'Arquivo inexistente.'

variable_errors=dados['variable_errors']
variable_errors_val=dados['variable_errors_val']
time=dados['time']
summary_writing_period=dados['summary_writing_period']
#validation_period=dados['validation_period']
batch_size=dados['batch_size']

 
#train
batch_number = range(0,len(variable_errors)*summary_writing_period,summary_writing_period)
plt.figure(1)
plt.subplot(111)
plt.plot(batch_number, variable_errors, 'b')
axes = plt.gca()
axes.set_ylim([0,1])
plt.title('Train')
plt.grid(True)

dkeys=dados.keys()
color_cycle = ["red", "blue", "yellow", "green", "black", "purple", "turquoise", "magenta", "orange", "chartreuse"]
for ft_key, ft_ind in zip(config.features_list, xrange(len(config.features_list))):
	ft_dic = [k for k in dkeys if (ft_key+"_") in k]
	ft_dic.sort()
	if len(ft_dic)>0:
		plt.figure(ft_ind+2)
		plt.grid(True)
		plt.suptitle(ft_key)
		axes = plt.gca()
		min_val=1
		max_val=0
		batch_number = range(0,len(dados[ft_dic[0]]))
		num_plots=min(10,len(ft_dic))	
		for ft_ch_key, ch in zip(ft_dic, xrange(len(ft_dic))):			
			if ch%10==0:			
				plt.subplot(math.ceil(len(ft_dic)/10.0),1,1+ch/10.0)
				plt.gca().set_color_cycle(color_cycle)
			ch_actvs=dados[ft_ch_key]
			min_val=min(min_val,min(ch_actvs))
			max_val=max(max_val,max(ch_actvs))
			plt.plot(batch_number, ch_actvs, label=str(ch).zfill(len(str(len(ft_dic)))))
			plt.legend()	
			print("feature map %s, channel%d: avegare %f, variance %f"%(ft_key, ch, np.mean(ch_actvs), np.var(ch_actvs)))
		diff=max_val-min_val		
		axes.set_ylim([min_val-0.05*diff,max_val+0.05*diff])	
				

#for key in dados.keys() if "conv" in key:
#	print key

#validation
#batch_number_val = range(validation_period,(len(variable_errors_val)+1)*validation_period,validation_period)
#plt.subplot(212)
#plt.plot(batch_number_val, variable_errors_val, 'r')
#axes = plt.gca()
#axes.set_ylim([0,1])
#plt.title('Validation')
#plt.grid(True)

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
