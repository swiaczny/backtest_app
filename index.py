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
from app import Cache, Content, ContentRight, ContentLeft

#  THEMES: COSMO | LUX | SPACELAB | CYBORG | DARKLY
app = dash.Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.CYBORG], 
)

# DUE TO MULTI-PAGE LAYOUT
app.config.suppress_callback_exceptions = True
server = app.server
server.secret_key = os.environ.get('secret_key', 'secret')

# LAYOUT --------------------------------------------------------------------------------------------------------------- 

# APP LAYOUT
app.layout = html.Div(
	[
		Content().content_main()
	],
)

# CALLBACKS ------------------------------------------------------------------------------------------------------------
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

@app.callback(
	[
		Output('graph_div','children'),
	],
	[	
		Input('cache_sim1','data'),
		Input('cache_sim2','data'),
		Input('cache_bm','data'),
	],
)
def update_graph(cache_sim1, cache_sim2, cache_bm):

	return [ContentLeft().content_graph(cache_sim1=cache_sim1, cache_sim2=cache_sim2, cache_bm=cache_bm)]


@app.callback(
	[
		Output('table_div','children'),
	],
	[	
		Input('cache_sim1','data'),
		Input('cache_sim2','data'),
		Input('cache_bm','data'),
	],
)
def update_table(cache_sim1, cache_sim2, cache_bm):
	
	return [ContentLeft().content_table(cache_sim1=cache_sim1, cache_sim2=cache_sim2, cache_bm=cache_bm)]


@app.callback(
	[
		Output('sim1_tb_div','children'),
		Output('sim2_tb_div','children'),
	],
	[	
		Input('cache_sim1','data'),
		Input('cache_sim2','data'),
	],
)
def update_top_bottom(cache_sim1, cache_sim2):

	if cache_sim1:
		return ContentRight().content_top_bottom_all(cache_sim1=cache_sim1,cache_sim2=cache_sim2)
	else:
		raise PreventUpdate

if __name__ == '__main__':
	app.run_server(debug=True, use_reloader=True)
