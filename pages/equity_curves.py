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

dash.register_page(__name__, path='/equity_curves')

layout = html.Div(
	dbc.Container([
		dbc.Row([
			dbc.Col([
				dbc.Nav([
					dbc.NavItem(
						dcc.Dropdown(list_assets(), value='KGH', searchable=False, clearable=False, id='ec_asset')
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='Strategy', children=[
							dcc.Checklist(['EMD-LSTM'], value=['EMD-LSTM'], id='strategies', labelStyle={'display': 'block'}),
						]),
					),
				])
			])
		]),
		dbc.Row([
			dbc.Col([
				dcc.Graph(id='ec_graph'),
			])
		]),
	], fluid=True)
)

@callback(
	Output('ec_graph', 'figure'),
	Input('ec_asset', 'value'),
	Input('strategies', 'value')
)

def display_graph(ec_asset, strategies):

	df = pd.read_csv('history/history_' + ec_asset + '.csv')

	fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[1.0])

	fig.add_trace(go.Scatter(x=df['Date'], y=df['Equity'], name='Equity'), row=1, col=1)
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Buy & Hold'], name='Buy & Hold'), row=1, col=1)

	fig.update_xaxes(
		showline=True, linewidth=2, linecolor='gray', mirror=True, 
		gridcolor='white')
	fig.update_yaxes(
		showline=True, linewidth=2, linecolor='gray', mirror=True, 
		gridcolor='white',
		zeroline=True, zerolinewidth=1, zerolinecolor='gray')

#	fig.update_xaxes(showticklabels=True, row=1, col=1)
#	fig.update_xaxes(showticklabels=True, row=2, col=1)

	fig.update_layout(
		height=800,
		margin_t=10, margin_l=10, margin_r=20, margin_b=20, 
		paper_bgcolor='white', plot_bgcolor='whitesmoke',
		xaxis_rangeslider_visible=False,
		showlegend=False
	)

	return fig