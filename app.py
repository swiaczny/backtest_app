import pandas as pd

# dash libs
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# internal
import constants as con
from simulation import Simulation, AnalyseSim, ImportData
from util import Returns

class Cache():

    def __analyze_sim(self, date_range, sector=None, reb_freq=None, weighting=None, bm_ticker=None):
        '''
            running simulations and initialize AnalyseSim class
        '''
        start_date = date_range
        end_date = '2020-10-13'

        if bm_ticker is None:
            # RUN SIM BASED ON SIM-PARAMETERS (E.G. SUBSET ON SECTOR, ETC.)
            sim = Simulation(
                start_date=start_date, end_date=end_date, 
                sector=sector, 
                capital=10**7, 
                weighting=weighting, reb_freq=reb_freq
            )

            analyse = AnalyseSim(
                dates=sim.sim_dates, 
                tickers=sim.tickers, 
                asset_d_mv=sim.simulate(), 
                asset_d_returns=sim.returns
            )

        else:
            # BENCHMARK (TICKER) PERFORMANCE AND METRICS
            data = ImportData(start_date, end_date)
            prices = data.import_price_data(ticker_list=bm_ticker)
            data_dict = data.vectorize_data(prices)
            tr = Returns().compute_total_return_idx(data_dict['returns'])

            analyse = AnalyseSim(
                dates=data_dict['data_dates'], 
                tickers=data_dict['tickers'], 
                asset_d_mv=tr*10**7, 
                asset_d_returns=data_dict['returns']
            )

        return analyse

    def cache_data(self, date_range, sectors, reb_freqs, weightings):
        '''
            caching data 
        '''

        sim1 = self.__analyze_sim(date_range, sectors[0], reb_freqs[0], weightings[0])
        sim2 = self.__analyze_sim(date_range, sectors[1], reb_freqs[1], weightings[1])
        bmsim = self.__analyze_sim(date_range, bm_ticker=['^GSPC'])

        cache = []
        for idx, s in enumerate([sim1, sim2, bmsim]):
            
            # metrics
            anu_ret, anu_vol, ir, dd, no_of_assets = s.pf_metrics()
            
            # perf contribution
            top, bottom = s.top_bottom_contrib(5)
            
            cache.extend(
                [
                    {
                        'sector': sectors[idx], 
                        'weighting': weightings[idx], 
                        'reb_freq': reb_freqs[idx], 
                        'sim_tr': s.pf_tr_to_df().to_dict(),
                        'anu_ret': anu_ret,
                        'anu_vol': anu_vol,
                        'ir': ir,
                        'dd': dd,
                        'no_of_assets': no_of_assets,
                        'top': top,
                        'bottom': bottom,
                        'weights': s.d_asset_weights(),
                        'tickers':s.tickers,
                    }
                ]
            )

        return (cache[0], cache[1], cache[2])

    def content_cache(self, cache_id):

        return dcc.Store(id=cache_id, storage_type='session')
        # return dcc.Store(id=cache_id)

class ContentNavbar():

    def content_navbar(self):
        return dbc.NavbarSimple(
            [
                dbc.Col(dbc.NavItem(dcc.Link('Home', href='/home', id='link_home', style={'color':'white'}))),
                dbc.Col(dbc.NavItem(dcc.Link('Results', href='/results', id='link_results', style={'color':'white'}))),
                # dbc.NavItem(dcc.NavLink('Home', href='/home', id='link_home', active='exact')),
                # dbc.NavItem(dcc.NavLink('Results', href='/results', id='link_results', active='exact')),
            
            ],
            brand='Backtest Viewer',
            brand_href='#',
            color='primary',
            dark=True,
            fluid=True,
            # sticky='top',
            fixed='top'
        )


class ContentStatic(Cache, ContentNavbar):
    '''
        static content like caches and navbar
    '''
    def content_static(self):
        layout = html.Div(
            [
                dcc.Location(id='url', refresh=False),
                self.content_navbar(),
                self.content_cache(cache_id='cache_sim1'),
                self.content_cache(cache_id='cache_sim2'),
                self.content_cache(cache_id='cache_bm'),
            ],
        )

        return layout


class ContentMenu():
  
    def content_sim_parameters(self, header, sector_id, rebalance_id, weighting_id):

        layout = html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(header),
                        dbc.CardBody(
                            [
                                dbc.FormGroup(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Label('Sector:'),
                                                className='mt-mb-2'
                                                ),
                                                dbc.Col(
                                                    dbc.Label('Rebalancing Frequency:'),
                                                className='mt-mb-2'
                                                )
                                            ], 
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Select(id=sector_id,options=[
                                                        {'label': 'basic materials', 'value': 'basic materials'},
                                                        {'label': 'industrials', 'value': 'industrials'},
                                                        {'label': 'energy', 'value': 'energy'},
                                                        {'label': 'utilities', 'value': 'utilities'},
                                                        {'label': 'consumer cyclical', 'value': 'consumer cyclical'},
                                                        {'label': 'consumer defensive', 'value': 'consumer defensive'},
                                                        {'label': 'real estate', 'value': 'real estate'},
                                                        {'label': 'financial services', 'value': 'financial services'},
                                                        {'label': 'healthcare', 'value': 'healthcare'},
                                                        {'label': 'technology', 'value': 'technology'},
                                                        {'label': 'All (loading time up to 3 min)', 'value': 'ALL'},
                                                    ],
                                                    value='basic materials',
                                                    className='mt-mb-2'
                                                    ),
                                                ),
                                                dbc.Col(
                                                    dbc.Select(id=rebalance_id,options=[
                                                        {'label': 'monthly', 'value': 'M'},
                                                        {'label': 'quarterly', 'value': 'Q'},
                                                        {'label': 'semi-annually', 'value': 'S'},
                                                        {'label': 'yearly', 'value': 'Y'}
                                                    ],
                                                    value='M',
                                                    className='mt-mb-2'
                                                    ),
                                                ),
                                            ], 
                                        ),

                                        html.Br(),

                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Label('Weighting Method:'),
                                                className='mt-mb-2'
                                                ),
                                            ], 
                                        ),
                                        
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Select(id=weighting_id,options=[
                                                        {'label': 'equal-weighted', 'value': 'EW'},
                                                        {'label': 'vol-weighted', 'value': 'VOL'}
                                                    ],
                                                    value='EW',
                                                    className='mt-mb-2'
                                                    ),
                                                ),
                                            ] 
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        
        return layout

    def content_date_range(self, date_id):

        layout = html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardHeader('Set Simulation Period'),
                        dbc.CardBody(
                            [
                                dbc.FormGroup(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Select(id=date_id,options=[
                                                        {'label': '1 Year ', 'value': '2019-10-11'},
                                                        {'label': '2 Years', 'value': '2018-10-12'},
                                                        {'label': '3 Years', 'value': '2017-10-13'},
                                                        {'label': '5 Years', 'value': '2015-10-13'},
                                                        {'label': '7 Years', 'value': '2013-10-11'},
                                                        {'label': 'Max', 'value': '2011-03-30'}
                                                    ],
                                                    value='2019-10-11',
                                                    className='mt-mb-2'
                                                    ),
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    self.content_button(text = 'RUN SIMULATION', button_id = 'button_simulate', color = 'primary')
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        )

        return layout

    def content_button(self, text, button_id, color, block=True, size=None):
        return dbc.Button(text, id = button_id, color=color, block=block, size=size)


class ContentLeft():

    def __format(self, in_int, to_percent=True, add_sign=False):
        # ugly code
        if to_percent==True:
            if add_sign == True:
                if in_int >= 0:
                    return '+'+str(round(100*in_int,2))+' %'
                else:
                    return str(round(100*in_int,2))+' %'
            else:
                return str(round(100*in_int,2))+' %'
        else:
            return str(round(in_int,2))

    def __layout_graph(self, title, y_title=None, y_format_percent=False, legend=False):

        y_format = None

        if y_format_percent == True:
            y_format = '%p'

        return {

            'title': title,

            'showlegend': legend,
            'legend' : {
                'orientation': 'h',
                'y':-0.15
            },

            'plot_bgcolor': con.BG_COLOR,
            'paper_bgcolor': con.BG_COLOR,

            'font': {
                'color': con.FONT_COLOR,
                'family':con.FONT_FAMILY
            },

            'yaxis': {
                'title': y_title,
                'showgrid':True,
                'gridcolor':con.GRID_COLOR,
                'zeroline':False,
                'tickformat': y_format
            },

            'xaxis': {
                'showgrid':True,
                'gridcolor':con.GRID_COLOR,
                'zeroline':False,
            }

        }

    def __get_sim_title(self, cache_sim):
        '''
            returns name of simulation
        '''
        return cache_sim['sector'].upper() + ' (' + cache_sim['weighting'] +', ' + cache_sim['reb_freq'] + ')'

    def content_graph(self, cache_sim1=None, cache_sim2=None, cache_bm=None):

        if cache_sim1 is not None:

            cache = [cache_sim1, cache_sim2, cache_bm]
            line_color = [con.SIM1_RED, con.SIM2_BLUE, con.SIM_BM]

            # init figure with layout
            figure = go.Figure(layout=self.__layout_graph(title='',y_title = 'Gross Total Return',legend=True, y_format_percent=True))

            # add traces to figure
            for idx, c in enumerate(cache):
                
                if c['sector']:
                    # in case of cache_bm (i.e. S&P 500 ts) sector, wheigting is None
                    _name = self.__get_sim_title(c)
                else:
                    _name = 'S&P 500'

                dates = [*c['sim_tr']['date_'].values()]
                pf_tr = [*c['sim_tr']['pf_tr'].values()]
                
                figure.add_trace(
                    go.Scatter(
                        x=dates, 
                        y=pf_tr, 
                        mode='lines', 
                        name=_name,
                        line={'color':line_color[idx]}
                    )
                )
                # add annotations
                x_value = dates[-1]
                y_value = pf_tr[-1]
                y_str = round(100*(y_value-1),1)
                text = f'{y_str} %'
                
                figure.add_annotation(xref='x', yref='y', xanchor='right', yanchor='bottom', showarrow=False, x=x_value,y=y_value, text=text),
                
                # log scaling
                # figure.update_yaxes(type="log", range=[0,2])

            # add figure to graph object
            graph = dcc.Graph(config={'staticPlot':False}, figure=figure)

            return graph

    def content_table(self, cache_sim1=None, cache_sim2=None, cache_bm=None):
        
        if cache_sim1 is not None:
            
            name1 = cache_sim1['sector'].upper() + ' (' + cache_sim1['weighting'] +', ' + cache_sim1['reb_freq'] + ')'
            name2 = cache_sim2['sector'].upper() + ' (' + cache_sim2['weighting'] +', ' + cache_sim2['reb_freq'] + ')'
            name3 = 'S&P 500'

            table_header = [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(""), 
                            html.Th(name1, style={'color':con.SIM1_RED}),
                            html.Th(name2, style={'color':con.SIM2_BLUE}),
                            html.Th(name3, style={'color':'grey'}),
                        ]
                    )
                )
            ]

            row1 = html.Tr(
                [
                    html.Td('Return (p.a)'), 
                    html.Td(self.__format(in_int=cache_sim1['anu_ret'], to_percent=True, add_sign=True)), 
                    html.Td(self.__format(in_int=cache_sim2['anu_ret'], to_percent=True, add_sign=True)), 
                    html.Td(self.__format(in_int=cache_bm['anu_ret'], to_percent=True, add_sign=True)), 
                ]
            )
            
            row2 = html.Tr(
                [
                    html.Td('Vol (p.a)'), 
                    html.Td(self.__format(in_int=cache_sim1['anu_vol'], to_percent=True)), 
                    html.Td(self.__format(in_int=cache_sim2['anu_vol'], to_percent=True)), 
                    html.Td(self.__format(in_int=cache_bm['anu_vol'], to_percent=True)), 
                ]
            )

            row3 = html.Tr(
                [
                    html.Td('IR'), 
                    html.Td(self.__format(in_int=cache_sim1['ir'], to_percent=False)), 
                    html.Td(self.__format(in_int=cache_sim2['ir'], to_percent=False)), 
                    html.Td(self.__format(in_int=cache_bm['ir'], to_percent=False)), 
                ]
            )

            row4 = html.Tr(
                [
                    html.Td('Max DD'),
                    html.Td(self.__format(in_int=cache_sim1['dd'], to_percent=True)), 
                    html.Td(self.__format(in_int=cache_sim2['dd'], to_percent=True)), 
                    html.Td(self.__format(in_int=cache_bm['dd'], to_percent=True)), 
                ]
            )

            row5 = html.Tr(
                [
                    html.Td('No of Assets'),
                    html.Td(self.__format(in_int=cache_sim1['no_of_assets'], to_percent=False)), 
                    html.Td(self.__format(in_int=cache_sim2['no_of_assets'], to_percent=False)), 
                    html.Td(self.__format(in_int=500, to_percent=False)), 
                ]
            )

            table_body = [html.Tbody([row1, row2, row3, row4, row5])]

            table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True, dark=False)	

            return table

    def content_graph_weights(self, cache_sim):
        '''
            100% stacked area chart with weights
        '''        
        # init figure with layout
        _title = self.__get_sim_title(cache_sim)

        figure = go.Figure(layout=self.__layout_graph(title=_title ,y_title = 'Weight in %',legend=True, y_format_percent=True))

        dates = [*cache_sim['sim_tr']['date_'].values()]

        for idx,w in enumerate(cache_sim['weights']):

            figure.add_trace(
                go.Scatter(
                    x=dates, 
                    y=w, 
                    mode='lines', 
                    name=cache_sim['tickers'][idx],
                    stackgroup='one',
                )
            )

        graph = dcc.Graph(config={'staticPlot':False}, figure=figure)        
        
        return graph


class ContentRight():

    def __format(self, in_int, to_percent=True, add_sign=False):
        # ugly code
        if to_percent==True:
            if add_sign == True:
                if in_int >= 0:
                    return '+'+str(round(100*in_int,2))+' %'
                else:
                    return str(round(100*in_int,2))+' %'
            else:
                return str(round(100*in_int,2))+' %'
        else:
            return str(round(in_int,2))

    def __top_bottom_table(self, cache, top):

        if cache is not None:

            if top == True:
                data = cache['top']
                srt = False
            else:
                data = cache['bottom']
                srt = True

            df = pd.DataFrame(data).T

            df['Symbol'] = df.index
            df.reset_index(drop=True, inplace=True)
            df = df[['Symbol', 'contribution', 'asset_tr']]
            df.sort_values(by='contribution',ascending=srt, inplace=True)

            # format to str
            df['contribution'] = df['contribution'].apply(lambda x: self.__format(in_int=x, to_percent=True, add_sign=True))
            df['asset_tr'] = df['asset_tr'].apply(lambda x: self.__format(in_int=x, to_percent=True, add_sign=True))

        return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

    def content_top_bottom_all(self, cache_sim1, cache_sim2):
        
        style1={'color':con.SIM1_RED}
        style2={'color':con.SIM2_BLUE}

        sim1_top_bottom = [
            # header
            html.Th('TOP 5', style=style1),
            self.__top_bottom_table(cache=cache_sim1, top=True), 
            # header
            html.Th('BOTTOM 5', style=style1),
            self.__top_bottom_table(cache=cache_sim1, top=False)
            ]

        sim2_top_bottom = [
            # header
            html.Th('TOP 5', style=style2), 
            self.__top_bottom_table(cache=cache_sim2, top=True), 
            # header
            html.Th('BOTTOM 5', style=style2), 
            self.__top_bottom_table(cache=cache_sim2, top=False)
        ]

        return (sim1_top_bottom, sim2_top_bottom)

class Content(Cache, ContentMenu):

    def __init__(self):
        super().__init__()

    def content_home(self):
        layout = html.Div(
            [   
                html.Br(),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    self.content_sim_parameters(
                                        header='Set Paramters for Simulation 1',
                                        sector_id='sector1',
                                        rebalance_id='reb_freq1', 
                                        weighting_id='weighting1'
                                    ),
                                ),
                                dbc.Col(
                                    self.content_sim_parameters(
                                        header='Set Paramters for Simulation 2',
                                        sector_id='sector2',
                                        rebalance_id='reb_freq2', 
                                        weighting_id='weighting2'
                                    ),
                                ),
                                dbc.Col(
                                    self.content_date_range(date_id='date_range')
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        style={'vertical-align': 'top'}
        ),

        return layout

    def content_results(self):

        layout = html.Div(
            [   
                html.Div(
                    [
                        html.Br(),
                        dbc.Container(
                            [
                                dbc.Row(
                                    [   
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        [   
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col('Portfolio Returns', width=3),
                                                                    dbc.Col(
                                                                        self.content_button(
                                                                            text = 'REFRESH', 
                                                                            button_id = 'button_refresh', 
                                                                            color = 'secondary', 
                                                                            size='sm'
                                                                        ),
                                                                    width = 3
                                                                    ),
                                                                ],
                                                            justify='between'
                                                            ),
                                                        ],
                                                    ),
                                                    dbc.CardBody(
                                                        [

                                                        ],
                                                    id = 'graph_div',
                                                    ),
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader('Metrics'),
                                                    dbc.CardBody(
                                                        [

                                                        ],
                                                    id = 'table_div',
                                                    ),
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                                html.Br(),
                            ],
                        fluid=True
                        ),
                    ],
                style={'width': '59%', 'display': 'inline-block', 'vertical-align': 'top'},
                ),
                # html.Br(),
                html.Div(
                    [
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader('Return Contributions: Simulation 1'),
                                                dbc.CardBody(
                                                    [

                                                    ],
                                                id = 'sim1_tb_div',
                                                ),
                                            ],
                                        ),
                                    ],
                                width =6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader('Return Contributions: Simulation 2'),
                                                dbc.CardBody(
                                                    [

                                                    ],
                                                id = 'sim2_tb_div',
                                                ),
                                            ],
                                        ),
                                    ],
                                width =6
                                ),
                            ],
                        ),
                    ],
                style={'width': '39%', 'display': 'inline-block', 'vertical-align': 'top'},
                ),
                html.Div(
                    [
                        dbc.Container(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader('Asset Weights'),
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [

                                                                        ],
                                                                    id = 'weights_div1',
                                                                    ),
                                                                ],
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [

                                                                        ],
                                                                    id = 'weights_div2',
                                                                    ),
                                                                ],
                                                            ),

                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        fluid=True
                        ),
                    ],
                style={'width': '59%', 'display': 'inline-block', 'vertical-align': 'top'},
                ),
            ],
        )

        return layout
