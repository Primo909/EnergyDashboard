
import dash
import dash_core_components as dcc
from dash import html
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_csv('data.csv')
available_years = df['year'].unique()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('IST Energy Monitor - Dashboard 6'),

    html.Div('Visualization of total electricity consumption at IST over the last years'),

    html.Div([
        dcc.Dropdown(
            id='menu',
            options=[{'label': i, 'value': i} for i in available_years],
            value=2017
        ),
    ]),

    html.Div([
        dcc.Graph(id='yearly-data')
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),

])


@app.callback(
    dash.dependencies.Output('yearly-data', 'figure'),
    [dash.dependencies.Input('menu', 'value')])
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