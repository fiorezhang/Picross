#picross dataset
#copyright fiorezhang@sina.com

import numpy as np
from picross_generator import generator as gen

def load_data(row, col, visible, num):
	x = np.zeros((num, row, col), dtype='int16')
	y = np.zeros((num, 2, row, col), dtype='int16')
	
	for i in range(num):
		x[i], y[i, 0], y[i, 1] = gen(row, col, visible)
		
	return x, y

