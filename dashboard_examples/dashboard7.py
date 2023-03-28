
import dash
from dash import html
import dash_core_components as dcc
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

app = dash.Dash(external_stylesheets=external_stylesheets)
app.layout =html.Div([
    html.H2('IST Energy Yearly Consumption (kWh)'),
    html.Div([
        html.Div([
            html.H3('Table'),
            generate_table(df),
        ], className="six columns"),

        html.Div([
            html.H3('Graph'),
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
            )
        ], className="six columns"),
    ], className="row")
])

if __name__ == '__main__':
    app.run_server(debug=True)
