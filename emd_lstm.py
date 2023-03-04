import pandas as pd
import numpy as np

from urllib.request import urlretrieve

import emd

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler


def import_historical_data(asset):
	url = 'https://stooq.com/q/d/l/?s=' + asset + '&i=d'
	csv_file = ('datasets/' + asset + '.csv')
	urlretrieve(url, csv_file)

def decomposition(asset, split_point):
	df = pd.read_csv('datasets/' + asset + '.csv')
	subset = df[df['Date'] < split_point]
	imf_df = pd.DataFrame()
	imf = emd.sift.sift(subset['Close'])
	imf_columns = []
	for i in range(imf.shape[1]):
		imf_df['IMF ' + str(i+1)] = imf[:,i]
	imf_df.set_index(subset.index, inplace=True)
	df = pd.concat([df, imf_df], axis = 1)
	df.to_csv('datasets/' + asset + '.csv', index=False)
	
def sequence_split(series, lag, steps):
	print(series.shape)
	x_sequences = []
	y_sequences = []
	for i in range(lag, series.shape[0] - steps):
		x_sequences.append(series.iloc[i - lag: i])
		y_sequences.append(series.iloc[i: i+steps])
	x_sequences = np.array(x_sequences)
	y_sequences = np.array(y_sequences)
	print(x_sequences.shape)
	print(y_sequences.shape)
	return x_sequences, y_sequences

def list_components(asset):
	df = pd.read_csv('datasets/' + asset + '.csv')
	components = []
	for column in df.columns:
		if ('IMF' in column) and (len(column) == 5):
			components.append(column)
	return components	

def train_model(asset, component, split_point, lag, steps):

	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers import Dropout
	from keras.layers import Dense
	from keras.models import model_from_json

	df = pd.read_csv('datasets/' + asset + '.csv')
	df = df[df['Date'] < split_point]

	scaler = MinMaxScaler()
	array = scaler.fit_transform(df[[component]])
	array = array.reshape(-1)

	series = pd.Series(array)

	x_train, y_train = sequence_split(series, lag, steps)

	model = Sequential()
	model.add(LSTM(units=32, return_sequences=True, input_shape=(lag, 1)))
	model.add(Dropout(0.2))
	model.add(LSTM(units=32, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(steps))
	model.compile(loss='mae', optimizer='adam')
	model.summary()

	history = model.fit(x_train, y_train, epochs=20, validation_split=0.2)

	def show_learning_curve():

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'])
		plt.show()

	def save_learning_curve():

		table = pd.DataFrame()
		table['loss'] = history.history['loss']
		table['val_loss'] = history.history['val_loss']
		table['epochs'] = [epoch for epoch in range(1, len(table)+1)]
		table.to_csv('models/model ' + asset + ' ' + component + '.csv', index=False)

	def save_model():

		model_json = model.to_json()
		with open('models/model ' + asset + ' ' + component + '.json', 'w') as json_file:
			json_file.write(model_json)
		model.save_weights('models/model ' + asset + ' ' + component + '.h5')

#	show_learning_curve()
	save_learning_curve()
	save_model()



def test_model(asset, component, split_point, lag, steps):

	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers import Dropout
	from keras.layers import Dense
	from keras.models import model_from_json

	# Load model
	json_file = open('models/model ' + asset + ' ' + component + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights('models/model ' + asset + ' ' + component + '.h5')



def rolling_decomposition(asset, lag, steps):

	df = pd.read_csv('datasets/' + asset + '.csv')

	# 0 axis - time steps
	# 1 axis - components
	# 2 axis - samples



	x_shape = 1
	y_shape = 2
	z_shape = 3

	x_array = np.zeros(shape=(x_shape, y_shape, z_shape))

	print(x_array.shape)

def main():
	
	asset = 'KGH'
	lag = 90
	steps = 10
	split_point = '2020-01-01'

#	import_historical_data(asset)
#	decomposition(asset, split_point)

#	components = list_components(asset)
#	for component in components:
#		train_model(asset, component, split_point, lag, steps)
#		test_model(asset,component, split_point, lag, steps)

	rolling_decomposition(asset, lag, steps)



if __name__ == "__main__":
	main()