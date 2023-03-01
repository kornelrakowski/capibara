import pandas as pd
import numpy as np

from urllib.request import urlretrieve
import requests
from bs4 import BeautifulSoup

import emd

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime, timedelta



def import_dataset(asset):
	url = 'https://stooq.com/q/d/l/?s=' + asset + '&i=d'
	csv_file = ('datasets/' + asset + '.csv')
	urlretrieve(url, csv_file)



def indicators(asset):

	# Reading source file
	df = pd.read_csv('datasets/' + asset + '.csv')

	# Typical price (HLC)
	df['Typical price'] = ( df['High'] + df['Low'] + df['Close'] ) / 3

	periods = [10, 20, 50, 100, 200]
	for period in periods:
		df['SMA ' + str(period)] = df['Close'].rolling(period).mean()
	for period in periods:
		df['EMA ' + str(period)] = df['Close'].ewm(span=period, min_periods=period, adjust=False).mean()

	# SMA Ratios
	df['SMA 10/50 ratio'] = df['SMA 10'] / df['SMA 50']
	df['SMA 20/100 ratio'] = df['SMA 20'] / df['SMA 100']		
	df['SMA 50/200 ratio'] = df['SMA 50'] / df['SMA 200']

	# EMA Ratios
	df['EMA 10/50 ratio'] = df['EMA 10'] / df['EMA 50']
	df['EMA 20/100 ratio'] = df['EMA 20'] / df['EMA 100']
	df['EMA 50/200 ratio'] = df['EMA 50'] / df['EMA 200']

	# Trend indicator
	df['Trend 20'] = df['Close'] / df['SMA 20']
	df['Trend 50'] = df['Close'] / df['SMA 50']
	df['Trend 100'] = df['Close'] / df['SMA 100']
	df['Trend 200'] = df['Close'] / df['SMA 200']

	# MACD
	EMA_26 = df['Close'].ewm(alpha=2/26).mean()
	EMA_12 = df['Close'].ewm(alpha=2/12).mean()
	df['MACD'] = EMA_12 - EMA_26
	df['MACD Signal Line'] = df['MACD'].ewm(alpha=2/9).mean()
	df['MACD Histogram'] = df['MACD'] - df['MACD Signal Line']

	# RSI
	df.loc[df['Close'] > df['Close'].shift(1), 'Upward change'] = df['Close'] - df['Close'].shift(1)
	df.loc[df['Close'] <= df['Close'].shift(1), 'Upward change'] = 0
	df.loc[df['Close'] < df['Close'].shift(1), 'Downward change'] = df['Close'].shift(1) - df['Close']
	df.loc[df['Close'] >= df['Close'].shift(1), 'Downward change'] = 0
	upward_SMMA = df['Upward change'].ewm(alpha=1/14).mean()
	downward_SMMA = df['Downward change'].ewm(alpha=1/14).mean()
	relative_strength = upward_SMMA / downward_SMMA
	rsi = 100 - (100 / (1+ relative_strength))
	df['RSI'] = rsi

	# Bollinger Bands
	standard_deviation = df['Typical price'].rolling(20).std()
	df['Upper band'] = df['SMA 20'] + 2*standard_deviation
	df['Lower band'] = df['SMA 20'] - 2*standard_deviation
	df['Percent_b'] = (df['Close'] - df['Lower band']) / (df['Upper band'] - df['Lower band'])
	df['Bandwidth'] = (df['Upper band'] - df['Lower band']) / df['SMA 20']

	# Stochastic
	df['Stochastic %K'] = ((df['Close'] - df['Low'].rolling(10).min()) / (df['High'].rolling(10).max() - df['Low'].rolling(10).min())) * 100
	df['Stochastic %D'] = df['Stochastic %K'].rolling(3).mean()

	# Williams %R
	df['Williams %R'] = ( df['High'].rolling(14).max() - df['Close']) / ( df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * -100

	# CCI
	df['CCI'] = ( df['Typical price'] - df['Typical price'].rolling(20).mean() ) / ( abs(df['Typical price'] - df['Typical price'].rolling(20).mean()).mean() * 0.015 )

	# Aroon
	df['Aroon Up'] = 100 * df.High.rolling(25 + 1).apply(lambda x: x.argmax()) / 25
	df['Aroon Down'] = 100 * df.Low.rolling(25 + 1).apply(lambda x: x.argmin()) / 25

	df.to_csv('datasets/' + asset + '.csv', index=False)
	print('Indicators saved to file datasets/'+ asset +'.csv')

def signals(asset):

	# Reading source file
	df = pd.read_csv('datasets/' + asset + '.csv')

	# Buy/sell signals
	df['SMA 10/50 Signal'] = np.select(
		[(df['SMA 10/50 ratio'] > 1) & (df['SMA 10/50 ratio'].shift(1) < 1), (df['SMA 10/50 ratio'] < 1) & (df['SMA 10/50 ratio'].shift(1) > 1)], 
		[1, -1])
	df['SMA 20/100 Signal'] = np.select(
		[(df['SMA 20/100 ratio'] > 1) & (df['SMA 20/100 ratio'].shift(1) < 1), (df['SMA 20/100 ratio'] < 1) & (df['SMA 20/100 ratio'].shift(1) > 1)], 
		[1, -1])
	df['SMA 50/200 Signal'] = np.select(
		[(df['SMA 50/200 ratio'] > 1) & (df['SMA 50/200 ratio'].shift(1) < 1), (df['SMA 50/200 ratio'] < 1) & (df['SMA 50/200 ratio'].shift(1) > 1)], 
		[1, -1])
	df['EMA 10/50 Signal'] = np.select(
		[(df['EMA 10/50 ratio'] > 1) & (df['EMA 10/50 ratio'].shift(1) < 1), (df['EMA 10/50 ratio'] < 1) & (df['EMA 10/50 ratio'].shift(1) > 1)], 
		[1, -1])
	df['EMA 20/100 Signal'] = np.select(
		[(df['EMA 20/100 ratio'] > 1) & (df['EMA 20/100 ratio'].shift(1) < 1), (df['EMA 20/100 ratio'] < 1) & (df['EMA 20/100 ratio'].shift(1) > 1)], 
		[1, -1])
	df['EMA 50/200 Signal'] = np.select(
		[(df['EMA 50/200 ratio'] > 1) & (df['EMA 50/200 ratio'].shift(1) < 1), (df['EMA 50/200 ratio'] < 1) & (df['EMA 50/200 ratio'].shift(1) > 1)], 
		[1, -1])
	df['MACD Signal'] = np.select(
		[(df['Trend 20'] > 1) & ((df['MACD Histogram']>0) & (df['MACD Histogram'].shift(1)<0)) , 
			(df['Trend 20'] < 1) & ((df['MACD Histogram']<0) & (df['MACD Histogram'].shift(1)>0))], 
		[1, -1])
	df['RSI Signal'] = np.select(
		[(df['Trend 20'] > 1) & (df['RSI'] > 40) & (df['RSI'].shift(1) < 40), 
			(df['Trend 20'] < 1) & (df['RSI'] < 60) & (df['RSI'].shift(1) > 60)], 
		[1, -1])
	df['Bollinger Signal'] = np.select(
		[(df['Close'] < df['Lower band']) & (df['Close'].shift(1) > df['Lower band'].shift(1)), 
			(df['Close'] > df['Upper band']) & (df['Close'].shift(1) < df['Upper band'].shift(1))], 
		[1, -1])
	df['Stochastic Signal'] = np.select(
		[(df['Trend 20'] > 1) & ((df['Stochastic %D']>20) & (df['Stochastic %D'].shift(1)<20)), 
			(df['Trend 20'] < 1) & ((df['Stochastic %D']<80) & (df['Stochastic %D'].shift(1)>80))],
		[1, -1])
	df['Williams %R Signal'] = np.select(
			[(df['Williams %R'] > -80) & (df['Williams %R'].shift(1) < -80), 
			(df['Williams %R'] < -20) & (df['Williams %R'].shift(1) > -20)],
		[1, -1])
	df['CCI Signal'] = np.select(
		[(df['CCI']>-100) & (df['CCI'].shift(1)<-100), 
			(df['CCI']<100) & (df['CCI'].shift(1)>100)],
		[1, -1])
	df['Aroon Signal'] = np.select(
		[(df['Trend 50'] > 1) & (df['Aroon Up'] > 70) & (df['Aroon Up'].shift(1) < 70), 
			(df['Trend 50'] < 1) & (df['Aroon Down'] > 70) & (df['Aroon Down'].shift(1) < 30)],
		[1, -1])

	df.to_csv('datasets/' + asset + '.csv', index=False)
	print('Signals saved to file datasets/'+ asset +'.csv')

def candlesticks(asset):

	# Reading source file
	df = pd.read_csv('datasets/' + asset + '.csv')

	realbody = abs(df['Open'] - df['Close'])
	candle_range = df['High'] - df['Low']
	upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
	lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']

	# TREND REVERSAL PATTERNS
	df['White Marubozu'] = np.where(
		(df['Close'] > df['Open']) &
		(df['Close'] == df['High']) &
		(df['Open'] == df['Low'])
	, 1, 0)
	df['Black Marubozu'] = np.where(
		(df['Close'] < df['Open']) &
		(df['Close'] == df['Low']) &
		(df['Open'] == df['High'])
	, -1, 0)

	df['Bullish Engulfing'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Close'] > df['Open']) &
		(df['Close'].shift(1) < df['Open'].shift(1)) &
		(df['Close'] > df['Open'].shift(1)) &
		(df['Open'] < df['Close'].shift(1))
	, 1, 0)
	df['Bearish Engulfing'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Close'] < df['Open']) &
		(df['Close'].shift(1) > df['Open'].shift(1)) &
		(df['Close'] < df['Open'].shift(1)) &
		(df['Open'] > df['Close'].shift(1))
	, -1, 0)
	df['Bullish Harami'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Close'] > df['Open']) &
		(df['Close'].shift(1) < df['Open'].shift(1)) &
		(df['Close'] < df['Open'].shift(1)) &
		(df['Open'] > df['Close'].shift(1))
	, 1, 0)
	df['Bearish Harami'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Close'] < df['Open']) &
		(df['Close'].shift(1) > df['Open'].shift(1)) &
		(df['Close'] > df['Open'].shift(1)) &
		(df['Open'] < df['Close'].shift(1))
	, -1, 0)
	df['Tweezer Bottom'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Open'] < df['Close']) &
		(df['Open'].shift(1) < df['Close'].shift(1)) &
		(df['Low'] == df['Low'].shift(1))
	, 1, 0)
	df['Tweezer Top'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Open'] > df['Close']) &
		(df['Open'].shift(1) > df['Close'].shift(1)) &
		(df['High'] == df['High'].shift(1))
	, -1, 0)
	df['Piercing Line'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Close'] > df['Open']) &
		(df['Close'].shift(1) < df['Open'].shift(1)) &
		(df['Open'] < df['Close'].shift(1)) &
		(df['Close'] > (df['Open'].shift(1) + df['Close'].shift(1))/2)
	, 1, 0)
	df['Dark Cloud Cover'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Close'] < df['Open']) &
		(df['Close'].shift(1) > df['Open'].shift(1)) &
		(df['Close'] > df['Open'].shift(1)) &
		(df['Close'] < (df['Open'].shift(1) + df['Close'].shift(1))/2)
	, -1, 0)

	df['Morning Star'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Close'].shift(2) < df['Open'].shift(2)) &
		(df['Close'] > df['Open']) &
		(df['Open'] > df[['Close', 'Open']].max(axis=1).shift(1)) &
		(df['Close'].shift(2) > df[['Close', 'Open']].max(axis=1).shift(1)) &
		(df['Close'] > (df['Close'].shift(2) + df['Open'].shift(2))/2)
	, 1, 0)
	df['Evening Star'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Close'].shift(2) > df['Open'].shift(2)) &
		(df['Close'] < df['Open']) &
		(df['Open'] < df[['Close', 'Open']].min(axis=1).shift(1)) &
		(df['Close'].shift(2) < df[['Close', 'Open']].min(axis=1).shift(1)) &
		(df['Close'] < (df['Close'].shift(2) + df['Open'].shift(2))/2)
	, -1, 0)
	df['Three White Soldiers'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Close'] > df['Open']) &
		(df['Close'].shift(1) > df['Open'].shift(1)) &
		(df['Close'].shift(2) > df['Open'].shift(2)) &
		(df['Close'] > df['Close'].shift(1)) &
		(df['Close'].shift(1) > df['Close'].shift(2)) &
		(df['Open'] > df['Open'].shift(1)) &
		(df['Open'].shift(1) > df['Open'].shift(2)) &
		(realbody > 0.8 * candle_range) &
		(realbody.shift(1) > 0.8 * candle_range.shift(1)) &
		(realbody.shift(2) > 0.8 * candle_range.shift(2)) &
		(df['Open'] < df['Close'].shift(1)) &
		(df['Open'].shift(1) < df['Close'].shift(2))
	, 1, 0)
	df['Three Black Crows'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Close'] < df['Open']) &
		(df['Close'].shift(1) < df['Open'].shift(1)) &
		(df['Close'].shift(2) < df['Open'].shift(2)) &
		(df['Close'] < df['Close'].shift(1)) &
		(df['Close'].shift(1) < df['Close'].shift(2)) &
		(df['Open'] < df['Open'].shift(1)) &
		(df['Open'].shift(1) < df['Open'].shift(2)) &
		(realbody > 0.8 * candle_range) &
		(realbody.shift(1) > 0.8 * candle_range.shift(1)) &
		(realbody.shift(2) > 0.8 * candle_range.shift(2)) &
		(df['Open'] < df['Close'].shift(1)) &
		(df['Open'].shift(1) < df['Close'].shift(2))
	, -1, 0)
	df['Three Inside Up'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Open'].shift(2) > df['Close'].shift(2)) &
		(df['Open'].shift(1) < df['Close'].shift(1)) &
		(df['Open'] < df['Close']) &
		(df['Close'].shift(1) < df['Open'].shift(2)) &
		(df['Close'].shift(2) < df['Open'].shift(1)) &
		(df['Close'] > df['Close'].shift(1)) &
		(df['Open'] > df['Open'].shift(1))
	,1 ,0)
	df['Three Inside Down'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Open'].shift(2) < df['Close'].shift(2)) &
		(df['Open'].shift(1) > df['Close'].shift(1)) &
		(df['Open'] > df['Close']) &
		(df['Close'].shift(1) > df['Open'].shift(2)) &
		(df['Close'].shift(2) > df['Open'].shift(1)) &
		(df['Close'] < df['Close'].shift(1)) &
		(df['Open'] < df['Open'].shift(1))
	,-1 ,0)
	df['Three Outside Up'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Open'].shift(2) > df['Close'].shift(2)) &
		(df['Open'].shift(1) < df['Close'].shift(1)) &
		(df['Open'] < df['Close']) &
		(df['Close'].shift(1) > df['Open'].shift(2)) &
		(df['Close'].shift(2) > df['Open'].shift(1)) &
		(df['Close'] > df['Close'].shift(1)) &
		(df['Open'] > df['Open'].shift(1))
	,1 ,0)
	df['Three Outside Down'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Open'].shift(2) < df['Close'].shift(2)) &
		(df['Open'].shift(1) > df['Close'].shift(1)) &
		(df['Open'] > df['Close']) &
		(df['Close'].shift(1) > df['Open'].shift(2)) &
		(df['Close'].shift(2) > df['Open'].shift(1)) &
		(df['Close'] < df['Close'].shift(1)) &
		(df['Open'] < df['Open'].shift(1))
	,-1 ,0)

	df['Upside Tasuki Gap'] = np.where(
		(df['Trend 20'] > 1) &
		(df['Close'].shift(2) > df['Open'].shift(2)) &
		(df['Close'].shift(1) > df['Open'].shift(1)) &
		(df['Open'] > df['Close']) &
		(df['Open'].shift(1) > df['Close'].shift(2)) &
		(df['Open'] > df['Open'].shift(1)) &
		(df['Close'] < df['Close'].shift(2)) &
		(df['Open'] < df['Close'].shift(1)) &
		(df['Close'] > df['Open'].shift(2))
	, 1, 0)
	df['Downside Tasuki Gap'] = np.where(
		(df['Trend 20'] < 1) &
		(df['Close'].shift(2) < df['Open'].shift(2)) &
		(df['Close'].shift(1) < df['Open'].shift(1)) &
		(df['Open'] < df['Close']) &
		(df['Open'].shift(1) < df['Close'].shift(2)) &
		(df['Open'] < df['Open'].shift(1)) &
		(df['Close'] > df['Close'].shift(2)) &
		(df['Open'] > df['Close'].shift(1)) &
		(df['Close'] < df['Open'].shift(2))
	, -1, 0)

	df.to_csv('datasets/' + asset + '.csv', index=False)
	print('Candlestick patterns saved to file datasets/'+ asset +'.csv')

def emd_decomposition(asset):
	df = pd.read_csv('datasets/' + asset + '.csv')

	imf = emd.sift.sift(df['Close'])
	print(imf.shape)

	imf_columns = []
	for i in range(imf.shape[1]):
		df['IMF ' + str(i+1)] = imf[:,i]

	df.to_csv('datasets/' + asset + '.csv', index=False)
	print('EMD Components saved to file datasets/'+ asset +'.csv')

def list_emd_components(asset):
	df = pd.read_csv('datasets/' + asset + '.csv')
	all_columns = df.columns
	emd_components = []
	for column in all_columns:
		if ('IMF' in column) and (len(column) == 5):
			emd_components.append(column)
	return emd_components

def sequence_split(series, seq_len, steps):
	sequences = []
	for i in range(series.shape[0] - seq_len - steps):
		sequences.append(series.iloc[i: i+seq_len])
	sequences = np.array(sequences)
	return sequences

def emd_lstm_train(asset):

	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers import Dropout
	from keras.layers import Dense
	from keras.models import model_from_json

	df = pd.read_csv('datasets/' + asset + '.csv')

	steps = 5
	seq_len = 60

	# Append dataframe with new rows
	if df['Close'].shape[0] == df.shape[0]:
		for step in range(steps):
			previous_date = datetime.strptime(df.iloc[df.shape[0]-1]['Date'], '%Y-%m-%d').date()
			next_date = previous_date + timedelta(days=1)
			new_row = pd.DataFrame({'Date': str(next_date)}, index=[df.shape[0]])
			df = pd.concat([df, new_row])

	df.to_csv('datasets/' + asset + '.csv', index=False)

	emd_components = list_emd_components(asset)
	for component in emd_components:
		print(component)

		df = pd.read_csv('datasets/' + asset + '.csv')

		scaler = MinMaxScaler()

		df['x'] = scaler.fit_transform(df[[component]])
		df['y'] = df['x'].shift(-steps)

		split_point = '2020-01-01'

		train_df = df[df['Date'] < split_point]
		test_df = df[df['Date'] > split_point]

		x_train = sequence_split(train_df['x'], seq_len, steps)
		x_test = sequence_split(test_df['x'], seq_len, steps)

		y_train = train_df['y'].iloc[seq_len:]
		y_test = test_df['y'].iloc[seq_len:]

		model = Sequential()
		model.add(LSTM(units=32, return_sequences=True, input_shape=(seq_len, 1)))
		model.add(Dropout(0.2))
		model.add(LSTM(units=32, return_sequences=False))
		model.add(Dropout(0.2))
		model.add(Dense(1))
		model.compile(loss='mae', optimizer='adam')
		model.summary()

		history = model.fit(x_train, y_train, epochs=20, validation_split=0.2)

		def show_learning_curve(history):

			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])
			plt.ylabel('loss')
			plt.xlabel('epoch')
			plt.legend(['train', 'validation'])
			plt.show()

		table = pd.DataFrame()

		table['loss'] = history.history['loss']
		table['val_loss'] = history.history['val_loss']

		table['epochs'] = [epoch for epoch in range(1, len(table)+1)]

		table.to_csv('models/model ' + asset + ' ' + component + '.csv', index=False)

		# Save model to file
		model_json = model.to_json()
		with open('models/model ' + asset + ' ' + component + '.json', 'w') as json_file:
		    json_file.write(model_json)
		# Save weights to file
		model.save_weights('models/model ' + asset + ' ' + component + '.h5')
		print('Saved model to disk')

def emd_lstm_test(asset):

	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers import Dropout
	from keras.layers import Dense
	from keras.models import model_from_json

	df = pd.read_csv('datasets/' + asset + '.csv')

	steps = 5
	seq_len = 60

	emd_components = list_emd_components(asset)
	for component in emd_components:
		print(component)

		df = pd.read_csv('datasets/' + asset + '.csv')

		scaler = MinMaxScaler()

		df['x'] = scaler.fit_transform(df[[component]])
		df['y'] = df['x'].shift(-steps)

		split_point = '2020-01-01'

		train_df = df[df['Date'] < split_point]
		test_df = df[df['Date'] > split_point]

		x_train = sequence_split(train_df['x'], seq_len, steps)
		x_test = sequence_split(test_df['x'], seq_len, steps)

		y_train = train_df['y'].iloc[seq_len:]
		y_test = test_df['y'].iloc[seq_len:]

		# load json and create model
		json_file = open('models/model ' + asset + ' ' + component + '.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights('models/model ' + asset + ' ' + component + '.h5')
		print("Loaded model from disk")

		y_pred_raw = model.predict(x_test)

		y_pred_raw = scaler.inverse_transform(y_pred_raw)

		y_pred = []
		for i in y_pred_raw:
			y_pred.append(i[0])

		y_pred = pd.Series(y_pred)

		y_pred.index = df.index[-len(y_pred):]

		df = pd.read_csv('datasets/' + asset + '.csv')
		df[component + ' LSTM Pred'] = y_pred
		df.to_csv('datasets/' + asset + '.csv', index=False)

	# COMPOSE BACK
	df = pd.read_csv('datasets/' + asset + '.csv')
	
	emd_components = list_emd_components(asset)
	lstm_predictions = []
	for component in emd_components:
		lstm_predictions.append(component + ' LSTM Pred')

	df['Close LSTM Pred'] = df[lstm_predictions].sum(axis=1)

	df.to_csv('datasets/' + asset + '.csv', index=False)	

def emd_lstm_signals(asset):

	df = pd.read_csv('datasets/' + asset + '.csv')

	df['EMD-LSTM Signal'] = np.select(
		[(df['Close LSTM Pred'].diff().shift(1) < 0) & (df['Close LSTM Pred'].diff() > 0),
		(df['Close LSTM Pred'].diff().shift(1) > 0) & (df['Close LSTM Pred'].diff() < 0)],
		[1, -1])

	df.to_csv('datasets/' + asset + '.csv', index=False)	

def backtest(assets, strategy):

	# Definition makes code more readable
	signal = strategy + ' Signal'

	def list_trades(assets, strategy):

		trades = pd.DataFrame()

		for asset in assets:
			df = pd.read_csv('datasets/' + asset + '.csv')

			# Creating columns useful for further processing
			df['Asset'] = asset
			df['Next Day Date'] = df['Date'].shift(-1)
			df['Next Day Price'] = df['Open'].shift(-1)

			trades_partial = pd.DataFrame()

			# Creating subset with rows with events (entry and exit signals) occured
			long_entries = df.loc[(df[signal] == 1)]
			long_exits = df.loc[(df[signal] == -1)]

			# First exit must be after first entry, if not, first exit is removed
			if long_entries['Date'].iloc[0] > long_exits['Date'].iloc[0]:
				long_exits = long_exits.drop(long_exits.index[[0]])
			# Numbers of entries and exits must be equal, if not, last entry is removed
			if long_entries.shape[0] != long_exits.shape[0]:
				long_entries = long_entries.drop(long_entries.index[[-1]])

			trades_partial['Asset'] = long_entries['Asset'].iloc[:].values
			trades_partial['Position'] = 'Long'
			trades_partial['Entry Date'] = long_entries['Next Day Date'].iloc[:].values
			trades_partial['Entry Price'] = long_entries['Next Day Price'].iloc[:].values

			trades_partial['Exit Date'] = long_exits['Next Day Date'].iloc[:].values
			trades_partial['Exit Price'] = long_exits['Next Day Price'].iloc[:].values

			trades = trades.dropna()

			trades = pd.concat([trades, trades_partial], axis=0)

		trades.to_csv('trades.csv', index=False)												# Add strategy name to file name (but replace '/'!)

	def summary_trades():

		trades = pd.read_csv('trades.csv')

		trades['Gross Return'] = trades['Exit Price'] - trades['Entry Price']
		commission = 0.003
		trades['Return'] = trades['Gross Return'] - ( trades['Exit Price'] * commission + trades['Entry Price'] * commission )
		trades['Return %'] = trades['Return'] / trades['Entry Price'] * 100

		entry_date = pd.to_datetime(trades['Entry Date'])
		exit_date = pd.to_datetime(trades['Exit Date'])
		trades['Duration'] = (exit_date - entry_date).astype('timedelta64[h]') / 24

		start_date = trades['Entry Date'].min()
		end_date = trades['Entry Date'].max()
		total_duration = datetime.strptime(end_date, '%Y-%m-%d').date() - datetime.strptime(start_date, '%Y-%m-%d').date()

		winning_trades = len(trades.loc[trades['Return'] > 0])
		losing_trades = len(trades.loc[trades['Return'] < 0])

		win_ratio = winning_trades / trades.shape[0]
		loss_ratio = losing_trades / trades.shape[0]

		average_win = trades.loc[trades['Return'] > 0, 'Return %'].mean()
		average_loss = trades.loc[trades['Return'] < 0, 'Return %'].mean()

		risk_to_reward_ratio = - average_loss / average_win

		expectancy = ( (win_ratio * average_win) - (loss_ratio * average_loss) ) / 100

		standard_deviation = trades['Return %'].std()

		summary = {
		'Strategy': strategy,
		'Start Date': start_date,
		'End Date': end_date,
		'Total duration': total_duration,
		'Trades': trades.shape[0],
		'Winning trades': winning_trades,
		'Losing trades': losing_trades,
		'Win Ratio': win_ratio,
		'Loss Ratio': loss_ratio,
		'Best trade %': trades['Return %'].max(),
		'Worst trade %': trades['Return %'].min(),
		'Average win %': average_win,
		'Average loss %': average_loss,
		'Average return %': trades['Return %'].mean(),
		'Longest duration [days]': trades['Duration'].max(),
		'Shortest duration [days]': trades['Duration'].min(),
		'Average duration [days]': trades['Duration'].mean(),
		'Risk to Reward Ratio': risk_to_reward_ratio,
		'Expectancy Ratio': expectancy,
		'Standard deviation': standard_deviation
		}

		for i in summary:
			print('{}: {}'.format(i, summary[i]))

	list_trades(assets, strategy)
	summary_trades()

def equity_curve(asset, strategy):

	def preparation(asset, strategy):

		df = pd.read_csv('datasets/' + asset + '.csv')

		start_date = '2020-01-01'
		end_date = '2023-01-01'
		df_slice = df[(df['Date'] > start_date) & (df['Date'] < end_date)]
		df = df_slice

		history = pd.DataFrame()
		history['Date'] = df['Date']
		history['Signal'] = df[strategy + ' Signal']
		history['Next Day Open'] = df['Open'].shift(-1)
		history['Close'] = df['Close']

		history['Cash'] = None
		history['Stocks'] = None
 
		history.to_csv('history/history_' + asset + '.csv', index=False)												# Add strategy name to file name (but replace '/'!)

	preparation(asset, strategy)

	history = pd.read_csv('history/history_' + asset + '.csv')

	print(history.info())

	cash = 100
	stocks = 0

	for i in range(len(history)):

		history['Cash'].iloc[i] = cash
		history['Stocks'].iloc[i] = stocks

		# Entry long position
		if history['Signal'].iloc[i] == 1:
			stocks = cash / history['Next Day Open'].iloc[i]
			cash = 0

		# Exit long position
		if history['Signal'].iloc[i] == -1:
			cash = cash + stocks * history['Next Day Open'].iloc[i]
			stocks = 0

	history['Equity'] = history['Cash'] + history['Stocks'] * history['Close']

	first_close_price = history['Close'].iloc[0]
	history['Buy & Hold'] = 100 / first_close_price * history['Close']

	history.to_csv('history/history_' + asset + '.csv', index=False)												# Add strategy name to file name (but replace '/'!)

def shit():

	signal = strategy + ' Signal'

	def list_trades(assets, strategy):

		trades = pd.DataFrame()
		for asset in assets:
			source = pd.read_csv('datasets/' + asset + '.csv')
			df = source[['Date', 'Asset', signal, 'Close', 'Next Day Open']]
			trades_partial = df.loc[(df[signal] == -1) | (df[signal] == 1)]
			trades = pd.concat([trades, trades_partial], axis=0)

		trades = trades.sort_values(by='Date')
		trades = trades.dropna()
		trades.to_csv('trades.csv', index=False)

#	list_trades(assets, strategy)

	trades = pd.read_csv('trades.csv')

	cash = 10000
	minimal_transaction = 1600
	commission = 0.003

	portfolio = {}
	for asset in assets:
		portfolio[asset] = 0
	print(portfolio)

	for i in range(trades.shape[0]):
		symbol = trades.loc[i, 'Asset']
		next_day_price = trades.loc[i, 'Next Day Open']

		# Entry long position
		if (trades.loc[i, signal] == 1) and (cash >= minimal_transaction):
			portfolio[symbol] = max(cash*0.1, minimal_transaction) / next_day_price
			cash -= max(cash*0.1, minimal_transaction) * (1 + commission)

		# Exit long position
		if (trades.loc[i, signal] == -1) and (portfolio[symbol] != 0):
			cash += (portfolio[symbol] * next_day_price) * (1 - commission)
			portfolio[symbol] = 0

		trades.loc[i, 'Cash'] = cash

	trades.to_csv('trades.csv', index=False)


	portfolio_value = 0
	for symbol in assets:
		df = pd.read_csv('datasets/' + asset + '.csv')
		last_price = df['Close'].dropna().iloc[-1]
		portfolio_value += portfolio[symbol] * last_price

	print(portfolio)
	print(cash)
	print(portfolio_value)

	plt.plot(trades['Cash'])
	plt.show()

def main():
	
	assets = ['ACP', 'ALE', 'CCC', 'CDR', 'CPS', 'DNP', 'JSW', 'KGH', 'KRU', 'KTY', 'LPP', 'MBK', 'OPL', 'PCO', 'PEO', 'PGE', 'PKN', 'PKO', 'PZU', 'SPL']
#	for asset in assets:

#		import_dataset(asset)

#		indicators(asset)
#		candlesticks(asset)
#		signals(asset)

#		emd_decomposition(asset)

	assets = ['ACP', 'CCC', 'CDR', 'CPS', 'DNP', 'JSW', 'KGH', 'KRU', 'KTY', 'LPP', 'MBK', 'OPL', 'PEO', 'PGE', 'PKN', 'PKO', 'PZU', 'SPL']
	assets = ['ACP', 'CCC', 'CDR', 'CPS', 'DNP', 'JSW', 'KRU', 'KTY', 'LPP', 'MBK', 'OPL', 'PEO', 'PGE', 'PKN', 'PKO', 'PZU', 'SPL']

	for asset in assets:

		emd_lstm_train(asset)
		emd_lstm_test(asset)
		emd_lstm_signals(asset)

		equity_curve(asset, 'EMD-LSTM')
	
	backtest(assets, 'EMD-LSTM')

if __name__ == "__main__":
	main()

# TO DO
# Comparison with random walk
# Calculate Max Drawdown
# Calculate Sharpe Ratio
# Calculate Sortino Ratio
# Make multi asset portfolio simulation