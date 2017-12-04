#picross dataset
#copyright fiorezhang@sina.com

import numpy as np
from picross_generator import generator as gen

def load_data(row, col, visible, num):
	x = np.zeros((num, 2, row, col), dtype='int16')
	y = np.zeros((num, row, col), dtype='int16')
	
	for i in range(num):
		y[i], x[i, 0], x[i, 1] = gen(row, col, visible)
		
	return x, y

