import pandas as pd

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# internal
from app import Cache, ContentStatic, Content, ContentRight, ContentLeft

#  THEMES: COSMO | CYBORG | DARKLY | SLATE
app = dash.Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.CYBORG])

server = app.server

# LAYOUT --------------------------------------------------------------------------------------------------------------- 

# APP LAYOUT
app.layout = html.Div(
	[
		ContentStatic().content_static(),
		html.Br(),
		html.Br(),
		html.Br(),
		html.Div(
			[
				html.Div(Content().content_home(),id='page_home', style={'display':'none'}),
				html.Div([Content().content_results()],id='page_results', style={'display':'none'}),
			],
		),
	],
)

# CALLBACKS ------------------------------------------------------------------------------------------------------------

## PAGE NAVIGATION
@app.callback(
	[	
		Output('page_home', 'style'),
		Output('page_results', 'style')
	],
	[
		Input('url', 'pathname')
	]
)
def display_page(pathname):
	# alternative to real mulitpage setup (faster reload and results-page keeps state)
	if pathname == '/results':
		return {'display':'none'}, {'display':'block'}  
	else:
		return {'display':'block'}, {'display':'none'}  

## WRITING TO CACHE
# RINNING SIMS AND CACHING DATA ON CLIENT SIDE
@app.callback(
	[	
		Output('cache_sim1','data'),
		Output('cache_sim2','data'),
		Output('cache_bm','data'),

	],
	[
		Input('button_simulate','n_clicks'),
		Input('date_range','value'),
		Input('sector1','value'),
		Input('reb_freq1','value'),
		Input('weighting1','value'),
		Input('sector2','value'),
		Input('reb_freq2','value'),
		Input('weighting2','value'),
	]
)
def cache_data(n_clicks, date_range, sector1, reb_freq1, weighting1, sector2, reb_freq2, weighting2):
	'''
		running simulations and caching data for sim1, sim2 and benchmark (S&P 500)
	'''
	changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

	if 'button_simulate' in changed_id:
		# [sim1, sim2, s&p500]
		sectors = [sector1, sector2, None]
		reb_freqs = [reb_freq1, reb_freq2, None]
		weightings = [weighting1, weighting2, None]

		return Cache().cache_data(date_range, sectors, reb_freqs, weightings)
	
	else:
		raise PreventUpdate

## READING FROM CACHE
# UPDATING TWR GRAPH
@app.callback(
	[
		Output('graph_div','children'),
		Output('table_div','children'),
		Output('weights_div1','children'),
		Output('weights_div2','children'),
		Output('sim1_tb_div','children'),
		Output('sim2_tb_div','children'),
	],
	[	
		Input('button_refresh','n_clicks'),
		Input('cache_sim1','data'),
		Input('cache_sim2','data'),
		Input('cache_bm','data'),
	],
)
def update_left(n_clicks, cache_sim1, cache_sim2, cache_bm):

	if cache_sim1 is not None:
		changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
		if 'button_refresh' in changed_id:
			tr_graph = [ContentLeft().content_graph(cache_sim1=cache_sim1, cache_sim2=cache_sim2, cache_bm=cache_bm)]
			metrics = [ContentLeft().content_table(cache_sim1=cache_sim1, cache_sim2=cache_sim2, cache_bm=cache_bm)]
			weights1 = [ContentLeft().content_graph_weights(cache_sim=cache_sim1)]
			weights2 = [ContentLeft().content_graph_weights(cache_sim=cache_sim2)]
			ret_contributions = ContentRight().content_top_bottom_all(cache_sim1=cache_sim1,cache_sim2=cache_sim2)

			return tr_graph, metrics, weights1, weights2, ret_contributions[0], ret_contributions[1]
	else:
		raise PreventUpdate

if __name__ == '__main__':
	app.run_server(debug=True)
