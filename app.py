from dash import Dash, dcc, html, Input, Output
import dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

import pandas as pd
import os

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions = True)

app.layout = html.Div(
	dbc.Container([
		dbc.Row([
			dbc.Col([
				dbc.Nav([
					dbc.NavItem(
						html.H4('Capibara')
					),
					dbc.NavItem(
						dbc.Button(
							dcc.Link('Home', href='home')
						)
					),
					dbc.NavItem(
						dbc.Button(
							dcc.Link('Technical Analysis', href='technical_analysis')
						)
					),
					dbc.NavItem(
						dbc.Button(
							dcc.Link('Learning Curves', href='learning_curves')
						)
					),
					dbc.NavItem(
						dbc.Button(
							dcc.Link('Equity Curves', href='equity_curves')
						)
					),
				])
			])
		]),
		dbc.Row([
			dbc.Col([
				dash.page_container
			])
		]),
	], fluid=True)
)

if __name__ == '__main__':
	app.run_server(debug=True, use_reloader=True)