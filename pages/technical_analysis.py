from dash import Dash, dcc, html, Input, Output, callback
import dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

import pandas as pd
import os

def list_assets():
	datasets = os.listdir('datasets/')
	assets = []
	for dataset in datasets:
		assets.append(dataset.replace('.csv',''))
	return assets

def list_emd_components(asset):
	df = pd.read_csv('datasets/' + asset + '.csv')
	all_columns = df.columns
	emd_components = []
	for column in all_columns:
		if ('IMF' in column) and (len(column) == 5):
			emd_components.append(column)
	return emd_components

dash.register_page(__name__, path='/technical_analysis')

layout = html.Div(
	dbc.Container([
		dbc.Row([
			dbc.Col([
				dbc.Nav([
					dbc.NavItem(
						dcc.Dropdown(list_assets(), value='KGH', searchable=False, clearable=False, id='asset')
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='Main chart', children=[
							dcc.RadioItems(['Candlesticks', 'Close'], 'Close', id='price_chart', labelStyle={'display': 'block'}),
						]),
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='Overlays', children=[
							dcc.Checklist(['SMA 10', 'SMA 20', 'SMA 50', 'SMA 100', 'SMA 200', 'EMA 10', 'EMA 20', 'EMA 50', 'EMA 100', 'EMA 200', 'Bollinger'], value=['SMA 20'] , id='overlays', labelStyle={'display': 'block'}),
						]),
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='Oscillator 1', children=[
							dcc.RadioItems(['Volume', 'MACD', 'RSI', 'Stochastic', 'SMA Ratios', 'EMA Ratios', 'Williams %R', 'CCI', 'Aroon', 'EMD Components'], 'EMD Components' , id='oscillator1', labelStyle={'display': 'block'}),
						]),
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='Oscillator 2', children=[
							dcc.RadioItems(['Volume', 'MACD', 'RSI', 'Stochastic', 'SMA Ratios', 'EMA Ratios', 'Williams %R', 'CCI', 'Aroon', 'EMD Components'], 'RSI' , id='oscillator2', labelStyle={'display': 'block'}),
						]),
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='Signals', children=[
							dcc.Checklist(['SMA 10/50', 'SMA 20/100', 'SMA 50/200', 'EMA 10/50', 'EMA 20/100', 'EMA 50/200', 'MACD', 'RSI', 'Bollinger', 'Stochastic', 'Williams %R', 'CCI', 'Aroon', 'EMD-LSTM'], value=['EMD-LSTM'] , id='signals', labelStyle={'display': 'block'}),
						]),
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='Candlestick Patterns', children=[
							dcc.Checklist(['White Marubozu', 'Black Marubozu', 
								'Bullish Engulfing', 'Bearish Engulfing', 'Bullish Harami', 'Bearish Harami', 'Tweezer Bottom', 'Tweezer Top', 'Piercing Line', 'Dark Cloud Cover', 
								'Morning Star', 'Evening Star', 'Three White Soldiers', 'Three Black Crows',
								'Three Inside Up', 'Three Inside Down', 'Three Outside Up', 'Three Outside Down',
								'Upside Tasuki Gap', 'Downside Tasuki Gap'], 
								value=['Upside Tasuki Gap', 'Downside Tasuki Gap'] , id='candlestick_patterns', labelStyle={'display': 'block'}),
						]),
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='EMD Components', children=[
							dcc.Checklist(list_emd_components('KGH'), value=['IMF 1'], id='emd_components', labelStyle={'display': 'block'}),
						]),
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='Predictions', children=[
							dcc.Checklist(['EMD-LSTM'], value=['EMD-LSTM'], id='predictions', labelStyle={'display': 'block'}),
						]),
					),
				])
			])
		]),
		dbc.Row([
			dbc.Col([
				dcc.Graph(id='graph'),
			])
		]),
	], fluid=True)
)

@callback(
    Output('graph', 'figure'),
    Input('asset', 'value'),
    Input('price_chart', 'value'),
    Input('overlays', 'value'),
    Input('oscillator1', 'value'),
    Input('oscillator2', 'value'),
    Input('signals', 'value'),
    Input('candlestick_patterns', 'value'),
    Input('emd_components', 'value'),
    Input('predictions', 'value')
)

def display_graph(asset, price_chart, overlays, oscillator1, oscillator2, signals, candlestick_patterns, emd_components, predictions):

	df = pd.read_csv('datasets/' + asset + '.csv')

	fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

	# MAIN CHART
	if price_chart == 'Candlesticks':
		fig.add_trace(go.Candlestick(
			x=df['Date'],
			open=df['Open'],
			high=df['High'],
			low=df['Low'],
			close=df['Close'],
			increasing_line_color='black',
			decreasing_line_color='black',
			increasing_fillcolor='white',
			decreasing_fillcolor='black',
			increasing_line_width=1,
			decreasing_line_width=1,
			showlegend=False,
		), row=1, col=1)
	elif price_chart == 'Close':
		fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'), row=1, col=1)

	# OVERLAYS		
	if 'SMA 10' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 10'], name='SMA 10'), row=1, col=1)
	if 'SMA 20' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 20'], name='SMA 20'), row=1, col=1)
	if 'SMA 50' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 50'], name='SMA 50'), row=1, col=1)
	if 'SMA 100' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 100'], name='SMA 100'), row=1, col=1)
	if 'SMA 200' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 200'], name='SMA 200'), row=1, col=1)
	if 'EMA 10' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 10'], name='EMA 10'), row=1, col=1)
	if 'EMA 20' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 20'], name='EMA 20'), row=1, col=1)
	if 'EMA 50' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 50'], name='EMA 50'), row=1, col=1)
	if 'EMA 100' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 100'], name='EMA 100'), row=1, col=1)
	if 'EMA 200' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 200'], name='EMA 200'), row=1, col=1)
	if 'Bollinger' in overlays:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper band'], name='Upper band'), row=1, col=1)
		fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower band'], name='Lower band'), row=1, col=1)

	if 'EMD-LSTM' in predictions:
		fig.add_trace(go.Scatter(x=df['Date'], y=df['Close LSTM Pred'], name='Close LSTM Pred'), row=1, col=1)

	# OSCILLATORS
	def oscillator_update(oscillator_number, row_number):
		if oscillator_number == 'Volume':
			fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=row_number, col=1)
			fig.update_yaxes(title_text='Volume', row=row_number, col=1)
		elif oscillator_number == 'MACD':
			fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], line_color='blue', name='MACD'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD Signal Line'], line_color='red', name='MACD Signal Line'), row=row_number, col=1)
			fig.add_trace(go.Bar(x=df['Date'], y=df['MACD Histogram'], name='MACD Histogram'), row=row_number, col=1)
			fig.update_yaxes(title_text='MACD', row=row_number, col=1)
		elif oscillator_number == 'RSI':
			fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line_color='blue', name='RSI'), row=row_number, col=1)
			fig.update_yaxes(title_text='RSI', tickvals=[30, 70], row=row_number, col=1)
		elif oscillator_number == 'Stochastic':
			fig.add_trace(go.Scatter(x=df['Date'], y=df['Stochastic %K'], name='Stochastic %K'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['Stochastic %D'], name='Stochastic %D'), row=row_number, col=1)
			fig.update_yaxes(title_text='Stochastic', tickvals=[20, 80], row=row_number, col=1)
		elif oscillator_number == 'SMA Ratios':
			fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 5/20 ratio'], name='SMA 5/20 ratio'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 10/50 ratio'], name='SMA 10/50 ratio'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 20/100 ratio'], name='SMA 20/100 ratio'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA 50/200 ratio'], name='SMA 50/200 ratio'), row=row_number, col=1)
			fig.update_yaxes(title_text='SMA Ratios', row=row_number)
		elif oscillator_number == 'EMA Ratios':
			fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 5/20 ratio'], name='EMA 5/20 ratio'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 10/50 ratio'], name='EMA 10/50 ratio'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 20/100 ratio'], name='EMA 20/100 ratio'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA 50/200 ratio'], name='EMA 50/200 ratio'), row=row_number, col=1)
			fig.update_yaxes(title_text='EMA Ratios', row=row_number)
		elif oscillator_number == 'Williams %R':
			fig.add_trace(go.Scatter(x=df['Date'], y=df['Williams %R'], name='Williams %R'), row=row_number, col=1)
			fig.update_yaxes(title_text='Williams %R', row=row_number)
		elif oscillator_number == 'CCI':
			fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI'], name='CCI'), row=row_number, col=1)
			fig.update_yaxes(title_text='CCI', tickvals=[-100, 0, 100], row=row_number)
		elif oscillator_number == 'Aroon':
			fig.add_trace(go.Scatter(x=df['Date'], y=df['Aroon Up'], name='Aroon Up'), row=row_number, col=1)
			fig.add_trace(go.Scatter(x=df['Date'], y=df['Aroon Down'], name='Aroon Down'), row=row_number, col=1)
			fig.update_yaxes(title_text='Aroon', row=row_number)
		elif oscillator_number == 'EMD Components':
			for component in emd_components:
				if component == list_emd_components('KGH')[-1]:
					fig.add_trace(go.Scatter(x=df['Date'], y=df[component], name=component), row=1, col=1)
				else:
					fig.add_trace(go.Scatter(x=df['Date'], y=df[component], name=component), row=row_number, col=1)
					fig.update_yaxes(title_text='EMD Components', row=row_number)
			if 'EMD-LSTM' in predictions:
				for component in emd_components:
					if component == list_emd_components('KGH')[-1]:
						fig.add_trace(go.Scatter(x=df['Date'], y=df[component + ' LSTM Pred'], name=component + ' LSTM Pred'), row=1, col=1)
					else:
						fig.add_trace(go.Scatter(x=df['Date'], y=df[component + ' LSTM Pred'], name=component + ' LSTM Pred'), row=row_number, col=1)
						fig.update_yaxes(title_text='EMD Components', row=row_number)

	oscillator_update(oscillator1, 2)
	oscillator_update(oscillator2, 3)

	# SIGNALS
	def show_signals(signal_type):
		df_slice = df.loc[df[signal_type + ' Signal'] == 1]
		x_list = df_slice['Date'].to_list()
		y_list = df_slice['Typical price'].to_list()
		fig.add_trace(go.Scatter(x=x_list, y=y_list, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name=signal_type)) 
		df_slice = df.loc[df[signal_type + ' Signal'] == -1]
		x_list = df_slice['Date'].to_list()
		y_list = df_slice['Typical price'].to_list()
		fig.add_trace(go.Scatter(x=x_list, y=y_list, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name=signal_type)) 

	for signal in signals:
		show_signals(signal)

	# CANDLESTICK PATTERS
	def show_candlestick_patterns(pattern):
		df_slice = df.loc[df[pattern] == 1]
		x_list = df_slice['Date'].to_list()
		y_list = df_slice['Typical price'].to_list()
		fig.add_trace(go.Scatter(x=x_list, y=y_list, mode='markers', marker=dict(color='green', symbol='circle', size=10), name=pattern)) 
		df_slice = df.loc[df[pattern] == -1]
		x_list = df_slice['Date'].to_list()
		y_list = df_slice['Typical price'].to_list()
		fig.add_trace(go.Scatter(x=x_list, y=y_list, mode='markers', marker=dict(color='red', symbol='circle', size=10), name=pattern)) 

	for pattern in candlestick_patterns:
		show_candlestick_patterns(pattern)

	# Removing empty dates from X axis            ! NEED RUN FASTER
#	dt_all = pd.date_range(start=df['Date'].iloc[0],end=df['Date'].iloc[-1])
#	dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df['Date'])]
#	dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
#	fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

	fig.update_xaxes(
		showline=True, linewidth=2, linecolor='gray', mirror=True, 
		gridcolor='white')
	fig.update_yaxes(
		showline=True, linewidth=2, linecolor='gray', mirror=True, 
		gridcolor='white',
		zeroline=True, zerolinewidth=1, zerolinecolor='gray')

	fig.update_xaxes(showticklabels=True, row=1, col=1)
	fig.update_xaxes(showticklabels=True, row=2, col=1)

	fig.update_layout(
		height=800,
		margin_t=10, margin_l=10, margin_r=20, margin_b=20, 
		paper_bgcolor='white', plot_bgcolor='whitesmoke',
		xaxis_rangeslider_visible=False,
		showlegend=False
	)

	return fig