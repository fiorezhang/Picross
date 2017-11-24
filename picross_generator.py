#picross_generator
#copyright: fiorezhang@sina.com

import numpy as np

#calculate continued '1' for a string
def calculate(s):
	g = s.shape[0]
	p = np.zeros((g), dtype='int16')
	x, y = 0, 0#x: current position in return string, y: whether in '1' slice
	for i in range(g):
		if y == 0:#not in '1' slice
			if s[i] == 1:
				y = 1
				p[x] += 1
		else:#in '1' slice
			if s[i] == 1:
				p[x] +=1
			else:
				x += 1
				y = 0
	return p

#generate the matrix, and calculated continued '1' number for rows and columns
def generator(row, col, intense):
	m = np.random.rand(row, col)
	m = np.where(m<intense, 1, 0)
	#print('Expect: ', intense, 'Actual: ', m.sum()/(row*col))
	#print(m)
	
	n_r = np.zeros((row, col), dtype='int16')
	n_c = np.zeros((row, col), dtype='int16')
	
	for i in range(row):
		n_r[i] = calculate(m[i])
	
	for j in range(col):
		n_c[:,j] = calculate(m[:,j])
	
	if (m.sum() != n_r.sum()) or (m.sum() != n_c.sum()):
		print('ERROR')
	
	return m, n_r, n_c
	
#test functions
'''
x, y, z = generator(10, 10, 0.6)
print(x)
print(y)
print(z)
'''