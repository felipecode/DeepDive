import numpy as np
import matplotlib.pyplot as plt
import time
import math
from config import *
from random import *
import colorsys
import os
import json
import glob

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

dados=[]
names=[]

json_files=sorted(glob.glob(config.summary_path+"/*.json"))

for f in json_files:
 outfile=open(f,'r')
 dados.append(json.load(outfile))
 name, _ = os.path.splitext(os.path.basename(f))
 names.append(name)

variable_errors=[]
variable_errors_val=[]
time=[]
summary_writing_period=[]
batch_size=[]

for i in xrange(len(dados)):
 variable_errors.append(dados[i]['variable_errors'])
 variable_errors_val.append(dados[i]['variable_errors_val'])
 time.append(dados[i]['time'])
 summary_writing_period.append(dados[i]['summary_writing_period'])
 batch_size.append(dados[i]['batch_size'])

color_cycle = ["blue", "red", "yellow", "green", "black", "purple", "turquoise", "magenta", "orange", "chartreuse"]
 
#train
plt.figure(1)
plt.subplot(111)
axes = plt.gca()
axes.set_ylim([0,1])
plt.title('Train')
plt.grid(True)

plt.gca().set_color_cycle(color_cycle)
for i in xrange(len(dados)):
 batch_number = range(0,len(variable_errors[i])*summary_writing_period[i],summary_writing_period[i])
 plt.plot(batch_number, variable_errors[i], label=names[i])
plt.legend()

if len(dados)>0:
	dkeys=dados[0].keys()

for ft_key, ft_ind in zip(config.features_list, xrange(len(config.features_list))):
	ft_dic = [k for k in dkeys if (ft_key+"_") in k]
	ft_dic.sort()
	if len(ft_dic)>0:
		plt.figure(ft_ind+2)
		plt.grid(True)
		plt.suptitle(ft_key)
		axes = plt.gca()
		axes.set_xlabel('Channel')
		axes.set_ylabel('Activation')
		num_plots=min(10,len(ft_dic))	
		for ft_ch_key, ch in zip(ft_dic, xrange(len(ft_dic))):			
			plt.gca().set_color_cycle(color_cycle)
			for i in xrange(len(dados)):
				ch_actvs=dados[i][ft_ch_key]
				if ch==0:
					plt.plot(np.full(len(ch_actvs),ch,dtype=int), ch_actvs, '.', label=names[i])
				else:
					plt.plot(np.full(len(ch_actvs),ch,dtype=int), ch_actvs, '.')
							
plt.legend(numpoints=1)			
plt.show()
