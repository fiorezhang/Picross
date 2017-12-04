#picross_rnn.py
#copyright: fiorezhang@sina.com

import numpy as np
from keras.models import Sequential
from keras import layers
from picross_dataset import load_data as load

#global variables for dataset
SIZE_ROW = 3
SIZE_COL = 5
VISIBLE = 0.6
TRAIN_SIZE = 5000
TEST_SIZE = 500

#generate data from picross generator/dataset functions
print('-'*50)
print('Generating data...')
print('Picross size: ', SIZE_ROW, 'x', SIZE_COL, ', visible: ', VISIBLE*100, '%')
print('Train samples: ', TRAIN_SIZE, ', test samples: ', TEST_SIZE)
x, y = load(SIZE_ROW, SIZE_COL, VISIBLE, TRAIN_SIZE+TEST_SIZE)
x_train, x_test = x[:TRAIN_SIZE], x[TRAIN_SIZE:]
y_train, y_test = y[:TRAIN_SIZE], y[TRAIN_SIZE:]
print(x.shape)
print(y.shape)
print(x_train.shape)
print(y_train.shape)
print(x[0])
print(y[0])

#set parameters for RNN modle
RNN=layers.SimpleRNN
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
ITERATION = 100
EPOCHS = 10

#build the model
print('-'*50)
print('Building model...')
model = Sequential()
#reshape input, connect 2 matrix and flatten
model.add(layers.Reshape((2*SIZE_ROW*SIZE_COL, 1), input_shape=(2, SIZE_ROW, SIZE_COL)))
#print(model.output_shape)
#import data to a RNN
model.add(RNN(HIDDEN_SIZE, dropout=0.1))
#print(model.output_shape)
#repeat r*c times for output
model.add(layers.RepeatVector(SIZE_ROW*SIZE_COL))
#print(model.output_shape)
#rnn again, set return_sequences to output for all time pieces
for _ in range(LAYERS):
	model.add(RNN(HIDDEN_SIZE, return_sequences=True, dropout=0.1))
#print(model.output_shape)
#flatten, 3D->2D
model.add(layers.TimeDistributed(layers.Dense(1)))
#print(model.output_shape)
#finally binary cross entropy
model.add(layers.Activation('sigmoid'))
#print(model.output_shape)
#model.add(layers.Reshape((SIZE_ROW, SIZE_COL)))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#print(model.output_shape)
model.summary()

#train the model and print information during the process
print('-'*50) 
print('Training...')
for iteration in range(1, ITERATION):
	print()
	print('-'*50)
	print('Iteration', iteration)
	model.fit(x_train, y_train.reshape(-1, SIZE_ROW*SIZE_COL, 1),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_test, y_test.reshape(-1, SIZE_ROW*SIZE_COL, 1)))	
	#show result in the middle
	for i in range(1): 
		ind = np.random.randint(0, len(x_test))
		rowx, rowy = x_test[np.array([ind])], y_test[np.array([ind])]
		preds = model.predict_classes(rowx, verbose=0)
		question = rowx[0]
		#print(question.shape)
		correct = rowy[0]
		#print(correct.shape)
		#print(preds.shape)
		guess = preds[0].reshape(SIZE_ROW, SIZE_COL)	
		print('Q','- '*25)
		print(question)
		print('A','- '*25)
		print(correct)
		if (correct == guess).all():
			print('Y','- '*25)
		else:
			print('N','- '*25)
		print(guess)
		print('- '*25)		  