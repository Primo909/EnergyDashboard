import dash
import dash_core_components as dcc
from dash import html

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv('IST_Total_Hourly_Model.csv')


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('IST Energy Monitor - Dashboard 5'),

    html.Div('Visualization of hourly electricity consumption at IST over the last years'),

    dcc.Graph(
        id='yearly-data',
        figure={
            'data': [
                {'x': df['Date'], 'y': df['Power (kW) [Y]'], 'type': 'line', 'name': 'Power'},
                {'x': df['Date'], 'y': df['Temperature (C) [X]'], 'type': 'line', 'name': 'Temperature'},

            ],
            'layout': {
                'title': 'IST hourly electricity consumption (kWh)'
            }
        }
    ),


])

if __name__ == '__main__':
    app.run_server(debug=False)