import dash
import dash_core_components as dcc
from dash import html

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv('data.csv')

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('IST Energy Monitor - Dashboard 4'),

    html.Div('Visualization of total electricity consumption at IST over the last years'),
    dcc.Graph(
        id='yearly-data',
        figure={
            'data': [
                {'x': df.year, 'y': df.Total, 'type': 'bar', 'name': 'Total'},
                {'x': df.year, 'y': df.Civil, 'type': 'bar', 'name': 'Civil'},
                {'x': df.year, 'y': df.Central, 'type': 'bar', 'name': 'Central'},
                {'x': df.year, 'y': df.NorthTower, 'type': 'bar', 'name': 'North Tower'},
                {'x': df.year, 'y': df.SouthTower, 'type': 'bar', 'name': 'South Tower'},
            ],
            'layout': {
                'title': 'IST yearly electricity consumption (MWh)'
            }
        }
    ),

    html.H4('Summary Table'),
    generate_table(df)
])

if __name__ == '__main__':
    app.run_server(debug=True)