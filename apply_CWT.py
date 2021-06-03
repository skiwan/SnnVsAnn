import numpy as np
import os
import pywt

def average_signal(data, stepsize):
    y = data.shape[0]
    x = data.shape[1]
    if x % stepsize != 0:
        print(f'stepsize {stepsize} not appropriate for input data lenght')
        return data
    prepared = data.reshape((y,int(x/stepsize),stepsize))
    return np.mean(prepared, axis=-1)

current_wd = os.getcwd()

files = [
['BCI4_2a_A01T','BCI4_2a_A01E']
,['BCI4_2a_A02T','BCI4_2a_A02E']
,['BCI4_2a_A03T','BCI4_2a_A03E']
#,'BCI4_2a_A04T' T04 is corrupted due to some technical issues during recording
,['BCI4_2a_A05T','BCI4_2a_A05E']
,['BCI4_2a_A06T','BCI4_2a_A06E']
,['BCI4_2a_A07T','BCI4_2a_A07E']
,['BCI4_2a_A08T','BCI4_2a_A08E']
,['BCI4_2a_A09T','BCI4_2a_A09E']
,['BCI3_3a_k3b' ,'BCI3_3a_k3b']
,['BCI3_3a_k6b' ,'BCI3_3a_k6b']
,['BCI3_3a_l1b' ,'BCI3_3a_l1b']

]

configs = [
	[4,15,5],
	[19,30,5]
]

dt = 0.025

for f_e in files:
	for conf in configs:
		low_pass = conf[0]
		high_pass = conf[1]
		stepsize = conf[2]
		file_root = f_e[0]

		# load file
		# apply cwt (split and rejoin channels)
		# save file

