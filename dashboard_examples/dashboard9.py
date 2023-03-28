
import dash
import dash_core_components as dcc
from dash import html
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv('data.csv')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.Label('Dropdown'),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': '2017', 'value': 2017},
            {'label': '2018', 'value': 2018},
            {'label': '2019', 'value': 2019}
        ],
        value=2017
    ),



    html.Label('Radio Items'),
    dcc.RadioItems(
        id='radio',
        options=[
            {'label': '2017', 'value': 2017},
            {'label': '2018', 'value': 2018},
            {'label': '2019', 'value': 2019}
        ],
        value=2018
    ),

    html.Label('Text Input'),
    dcc.Input(value=2017, type='number',id='text'),

    html.Label('Slider'),
    dcc.Slider(
        min=2017,
        max=2019,
        marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(2017,2019)},
        value=2017,
        id='slider'
    ),



    html.Div([
        dcc.Graph(id='yearly-data')
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
], style={'columnCount': 1})

@app.callback(
    dash.dependencies.Output('yearly-data', 'figure'),
    [#dash.dependencies.Input('slider', 'value')]
     #dash.dependencies.Input('text', 'value')
     #dash.dependencies.Input('slider', 'value')
     #dash.dependencies.Input('dropdown', 'value')
     dash.dependencies.Input('radio', 'value')],
     

)



def update_graph(value):
    dff = df[df['year'] == value]
    return create_graph(dff)


def create_graph(dff):
    return {
        'data': [
            {'x': dff.year, 'y': dff.Total, 'type': 'bar', 'name': 'Total'},
            {'x': dff.year, 'y': dff.Civil, 'type': 'bar', 'name': 'Civil'},
            {'x': dff.year, 'y': dff.Central, 'type': 'bar', 'name': 'Central'},
            {'x': dff.year, 'y': dff.NorthTower, 'type': 'bar', 'name': 'North Tower'},
            {'x': dff.year, 'y': dff.SouthTower, 'type': 'bar', 'name': 'South Tower'},
        ],
        'layout': {
            'title': 'IST hourly electricity consumption (kWh)'
        }
    }




if __name__ == '__main__':
    app.run_server(debug=True)
