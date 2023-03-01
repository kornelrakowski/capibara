import schedule
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os

def list_assets():
	datasets = os.listdir('datasets/')
	assets = []
	for dataset in datasets:
		assets.append(dataset.replace('.csv',''))

	return assets

def get_ohlcv(asset):
	url = 'https://stooq.pl/q/a2/d/?s=' + asset + '&i=1'
	response = requests.get(url)
	soup = BeautifulSoup(response.content, 'html.parser')
	content = soup.get_text()
	daily_df = pd.DataFrame([x.split(',') for x in content.split('\n')], columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
	date = datetime.today().strftime('%Y-%m-%d')
	daily_open = daily_df['Open'].iloc[0]
	daily_high = daily_df['High'].max()
	daily_low = daily_df['Low'].min()
	daily_close = daily_df['Close'].iloc[-2]
	daily_volume = daily_df['Volume'].sum()

	df = pd.DataFrame(index=[0])

	df['Date'] = date
	df['Open'] = daily_open
	df['High'] = daily_high
	df['Low'] = daily_low
	df['Close'] = daily_close
	df['Volume'] = daily_volume

	print(df.head())

def update():

	assets = list_assets()

	for asset in assets:
		print(asset)
		get_ohlcv(asset)

update()

#schedule.every().day.at('18:00').do(update)

#while True:
#	schedule.run_pending()

