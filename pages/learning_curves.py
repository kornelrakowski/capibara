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

dash.register_page(__name__, path='/learning_curves')

layout = html.Div(
	dbc.Container([
		dbc.Row([
			dbc.Col([
				dbc.Nav([
					dbc.NavItem(
						dcc.Dropdown(list_assets(), value='KGH', searchable=False, clearable=False, id='lc_asset')
					),
					dbc.NavItem(
						dbc.DropdownMenu(label='EMD Components', children=[
							dcc.RadioItems(list_emd_components('KGH'), 'IMF 1', id='lc_emd_components', labelStyle={'display': 'block'}),
						]),
					),
				])
			])
		]),
		dbc.Row([
			dbc.Col([
				dcc.Graph(id='lc_graph'),
			])
		]),
	], fluid=True)
)

@callback(
	Output('lc_graph', 'figure'),
	Input('lc_asset', 'value'),
	Input('lc_emd_components', 'value')
)

def display_graph(lc_asset, lc_emd_components):

	df = pd.read_csv('models/model ' + lc_asset + ' ' + lc_emd_components + '.csv')

	fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[1.0])

	fig.add_trace(go.Scatter(x=df['epochs'], y=df['loss'], name='loss'), row=1, col=1)
	fig.add_trace(go.Scatter(x=df['epochs'], y=df['val_loss'], name='val_loss'), row=1, col=1)

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