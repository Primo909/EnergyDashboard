import dash
from dash import html,Input,Output,dcc
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pickle
from sklearn import metrics
import numpy as np
from datetime import date

featurelist_LR = ['Power_kW', 'solarRad_W/m2',
       'windGust_m/s', 'Holiday', 'Weekday', 'Hour', 'Month','Power -1']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# reading and formatting training data
north_data = pd.read_csv('IST_Kevin_Steiner_all_data.csv')
north_data['time'] = pd.to_datetime(north_data['time'], format="%Y-%m-%d %H:%M")
north_data = north_data.set_index('time', drop=True)


# reading and formatting test data 2019
aug_all = pd.read_csv('aug_all.csv')
aug_all['time'] = pd.to_datetime(aug_all['time'], format="%Y-%m-%d %H:%M")
aug_all = aug_all.set_index('time', drop=True)


# reading and formatting test data 2019
aug_2019 = pd.read_csv('project2_2019_aug.csv')
aug_2019['time'] = pd.to_datetime(aug_2019['time'], format="%Y-%m-%d %H:%M")
aug_2019 = aug_2019.set_index('time', drop=True)

# reading and formatting test data 2019
raw_2019 = pd.read_csv('project2_2019_raw.csv')
raw_2019['time'] = pd.to_datetime(raw_2019['time'], format="%Y-%m-%d %H:%M")
raw_2019 = raw_2019.set_index('time', drop=True)
#=============================
# define ML metrics
def ml_metrics(pred, check):
    MAE=metrics.mean_absolute_error(check,pred) 
    MBE=np.mean(check-pred) 
    MSE=metrics.mean_squared_error(check,pred)  
    RMSE= np.sqrt(metrics.mean_squared_error(check,pred))
    cvRMSE=RMSE/np.mean(check)
    NMBE=MBE/np.mean(check)
    r2 = metrics.r2_score(check,pred)
    d = [r2,
     MAE,
     MBE,
     MSE,
     RMSE,
     cvRMSE,
     NMBE]
    i = ['R2  ',
    'MAE ',
    'MBE ',
    'MSE ',
    'RMSE',
    'cvRM',
    'NMBE']
    df=pd.DataFrame({'Metric': i,'Values': d})
    print(df)
    return df


#=============================
# plot random forest prediction
#=============================
#Load LR model
#with open('final_model.pkl','rb') as file:
#    final_model=pickle.load(file)
#
#featurelist_final = ['Power_kW', 'HR', 'pres_mbar', 'rain_mm/h', 'solarRad_W/m2', 'temp_C',
#       'windGust_m/s', 'windSpeed_m/s', 'Holiday', 'Weekday', 'Hour', 'Month', 'Power -1']
#
#df = aug_all.loc['2019'][featurelist_final]
#
#t = df.index
#
#
#Z = df.values
#Y = df['Power_kW'].values
#X = df.drop(columns=['Power_kW']).values
#
#
#final_pred = final_model.predict(X)
#
#
#d={'time':t,'real-data':Y,'prediction':final_pred}
#df_forecast = pd.DataFrame(data=d)
###print(df_forecast)
#df_metrics = ml_metrics(final_pred,Y)
##
##randFig = px.line(df_forecast,
##        x=df_forecast.columns[0],
##        y=df_forecast.columns[1:]
##        )
##
###plt.subplot(1,2,1)
###plt.plot(t,Y)
###plt.plot(t,final_pred)
##
##plt.subplot(1,2,2)
##plt.scatter(Y,final_pred)
#
##ml_metrics(final_pred,Y)
##=============================



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H2('IST Energy Yearly Consumption (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Exploring the Initial Data', value='tab-1'),
        dcc.Tab(label='Model Evaluation', value='tab-2'),
    ]),
    html.Div(id='tabs-content'),

])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST Energy yearly Consumption (kWh)'),
            html.Label('Data availeable from 2017-01-01 to 2019-04-11'),
            #dcc.RadioItems(
            #    id='radio',
            #    options=[
            #        {'label': '2017', 'value': '2017'},
            #        {'label': '2018', 'value': '2018'},
            #        {'label': '2019', 'value': '2019'},
            #    ],
            #    value='2018',inline=True
            #),
            dcc.DatePickerRange(
                id='date-picker-raw',
                start_date=date(2017,1,1),
                end_date=date(2019,4,11)),
            dcc.Checklist(aug_all.columns,['Power_kW','temp_C'],id='features-explore',inline=True,),
            dcc.RadioItems(id='normalized',
                options=[
                {'label': 'Raw', 'value': 'notnorm'},
                {'label': 'Normalized', 'value': 'norm'},
                ],
                value='norm', inline=True),
            dcc.Graph(id='yearly-data'),
            ])
    elif tab == 'explore':
        return html.Div([
            html.H3('Choose')
            ])
    elif tab == 'tab-2':
        return html.Div([
            #generate_table(df_metrics),
            html.H3('Prediction for Early 2019 with a Chosen Model'),
            dcc.RadioItems(
                id='model-choose',
                options=[
                    {'label': 'Random Forest', 'value': 'RF'},
                    {'label': 'Linear Regression', 'value': 'LR'},
                ],
                value='RF',inline=True
            ),
            html.P(id='feature-list'),
            html.Div([dcc.Graph(id='random-forest')], style={'width': '70%', 'display': 'inline-block', 'padding': '0 20'}),
            #html.Div([generate_table(df_metrics)], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div(id='model-table', style={'width': '30%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
            
def choose_model(model):
    if model=='RF':
        filename='final_model.pkl'
        featurelist = ['Power_kW', 'HR', 'pres_mbar', 'rain_mm/h', 'solarRad_W/m2', 'temp_C', 'windGust_m/s', 'windSpeed_m/s', 'Holiday', 'Weekday', 'Hour', 'Month', 'Power -1']
    elif model=='LR':
        filename='linear_regr.pkl'
        featurelist = ['Power_kW', 'solarRad_W/m2', 'windGust_m/s', 'Holiday', 'Weekday', 'Hour', 'Month','Power -1']
    print(model)
    with open(filename,'rb') as file:
        model=pickle.load(file)
    
    df = aug_all.loc['2019'][featurelist]
    t = df.index
    
    Z = df.values
    Y = df['Power_kW'].values
    X = df.drop(columns=['Power_kW']).values
    
    pred = model.predict(X)
    
    d={'time':t,'real-data':Y,'prediction':pred}
    df_forecast = pd.DataFrame(data=d)

    df_metrics = ml_metrics(pred,Y)
    return df_forecast, df_metrics, featurelist
    
@app.callback(
        Output('random-forest','figure'),
        [Input('model-choose','value')])
def update_graph(model):
    df_forecast, dump,dump2 = choose_model(model)
    fig = px.line(df_forecast,
            x=df_forecast.columns[0],
            y=df_forecast.columns[1:]
            )
    return fig

@app.callback(
    Output('yearly-data', 'figure'),
    [#Input('radio', 'value'),
        Input('features-explore','value'),
        Input('normalized','value'),
        Input('date-picker-raw','start_date'),
        Input('date-picker-raw','end_date')])
def update_graph(featurelist,normalized,start,end):
    dff = aug_all.loc[start:end]

    if normalized=='norm':
        dff = dff.apply(lambda x: ((x - x.min()) / x.max()))
    fig = px.line(dff,
            x=dff.index,
            y=featurelist)
    return fig

@app.callback(
    Output('model-table', 'children'),
    [Input('model-choose','value')])
def model_table(model):
    df_pred, df_metrics,dump = choose_model(model) 
    return generate_table(df_metrics)
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

@app.callback(
        Output('feature-list', 'children'),
        [Input('model-choose', 'value')])
def show_featurelist(model):
    dump1,dump2,featurelist = choose_model(model)
    out = "This model uses the features: "+', '.join(featurelist)
    return html.P(out)

if __name__ == '__main__':
    #app.run_server(debug=True, port=8010)
    app.run_server(debug=True)
