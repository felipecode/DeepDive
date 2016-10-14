import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
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

#json_files=sorted(glob.glob(config.summary_path+"/*.json"))
json_files=sorted(glob.glob("../summary/*.json"))

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
	if ft_key in dados[0]:
		data_length=[]
		n_channels=np.array(dados[i][ft_key]).shape[1]
		actvs=np.empty([0,n_channels],dtype=np.float32)

		#means=np.empty([len(dados),n_channels],dtype=np.float32)
		#variances=np.empty([len(dados),n_channels],dtype=np.float32)

		fig = plt.figure() # Make a plotting figure
		ax = Axes3D(fig) # use the plotting figure to create a Axis3D object.	

		#min_vals=np.amin(np.array(pca_r.Y[:,0:3]),axis=0)
		#max_vals=np.amax(np.array(pca_r.Y[:,0:3]),axis=0)
 		

		# make simple, bare axis lines through space:
		#xAxisLine = ((min_vals[0], max_vals[0]), (0, 0), (0,0)) # 2 points make the x-axis line at the data extrema along x-axis 
		#ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'black') # make a red line for the x-axis.
		#yAxisLine = ((0, 0), (min_vals[1], max_vals[1]), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
		#ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'black') # make a red line for the y-axis.
		#zAxisLine = ((0, 0), (0,0), (min_vals[2], max_vals[2])) # 2 points make the z-axis line at the data extrema along z-axis
		#ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'black') # make a red line for the z-axis.

		# label the axes
		ax.set_xlabel("PC1") 
		ax.set_ylabel("PC2")
		ax.set_zlabel("PC3")
		ax.set_title(ft_key)

		for i in xrange(len(dados)):
			actvs=np.array(dados[i][ft_key])
			pca_r=PCA(actvs)
			x = []
			y = []
			z = []
			for item in pca_r.Y:
 				x.append(item[0])
 				y.append(item[1])
 				z.append(item[2])
			pltData = [x,y,z]
			ax.scatter(pltData[0], pltData[1], pltData[2], s=5, marker='.', color=color_cycle[i], edgecolors='none') # make a scatter plotfrom the data
			ax.plot([], [], 'o', color=color_cycle[i], label=names[i])#gambiarra pra legenda aparecer		

		plt.legend(numpoints=1)
		
		
		
											
plt.show()
