from dash import Dash, dcc, html, Input, Output, callback
from dash import dash_table
import dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

import pandas as pd
import os

dash.register_page(__name__, path='/')

def list_assets():
	datasets = os.listdir('datasets/')
	assets = []
	for dataset in datasets:
		assets.append(dataset.replace('.csv',''))
	return assets

assets = list_assets()

layout = html.Div(
	dbc.Container([
		dbc.Row([
			dbc.Col([
				dbc.Nav([
					dbc.NavItem(

					),
				])
			])
		]),
		dbc.Row([
			dbc.Col(
				
			)
		]),
	], fluid=True)
)

