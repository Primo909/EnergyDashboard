import dash
from dash import html,Input,Output,dcc
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pickle
from sklearn import metrics
import numpy as np
from datetime import date
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

featurelist_LR = ['Power_kW', 'solarRad_W/m2',
       'windGust_m/s', 'Holiday', 'Weekday', 'Hour', 'Month','Power -1']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



# reading and formatting test data 2019
aug_all = pd.read_csv('aug_all.csv')
aug_all['time'] = pd.to_datetime(aug_all['time'], format="%Y-%m-%d %H:%M")
aug_all = aug_all.set_index('time', drop=True)

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
    #print(df)
    return df

#=============================
# define split-features
def split_features(featurelist,split=0.75):
    df = aug_all.loc['2017':'2018']
    t = df.index
    Y = df['Power_kW'].values
    X = df[featurelist].values
    return X, Y, t
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
    html.H2('Energy Consuption of North Tower at IST (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Exploring the Initial Data', value='tab-1'),
        dcc.Tab(label='Model Evaluation', value='tab-2'),
        dcc.Tab(label='Train Your Model', value='train-model'),
    ]),
    html.Div(id='tabs-content'),

])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Raw Power Data and Raw Meteorological Data'),
            html.H6('Data availeable from 2017-01-01 to 2019-04-11'),
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
            html.Div(id='model-table', style={'width': '90%', 'display': 'inline-block', 'padding': '10 10'}),
            html.Div([dcc.Graph(id='random-forest')], style={'width': '60%', 'display': 'inline-block', 'padding': '10 10'}),
            html.Div([dcc.Graph(id='scatter-ml')], style={'width': '30%', 'display': 'inline-block', 'padding': '10 10'}),
            #html.Div([generate_table(df_metrics)], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    elif tab=='train-model':
        return html.Div([
            html.H3('Choose Features for your Model'),
            dcc.Checklist(aug_all.drop(columns=['Power_kW']).columns,['temp_C','windGust_m/s'],id='feature-checklist',inline=True,),
            html.H3('Choose your Model (Explanation below)'),
            dcc.RadioItems(
                id='own-model-choose',
                options=[
                    {'label': 'Linear Regression', 'value': 'LR'},
                    {'label': 'Random Forest', 'value': 'RF'},
                ],
                value='LR',inline=True
            ),
            html.H3('Train!'),
            html.Button('Train', id='submit-button'),
            html.Div(id='train-own-table'),
            html.Div([dcc.Graph(id='button-output')],style={'width': '60%', 'display': 'inline-block', 'padding': '10 10'}),
            html.Div([dcc.Graph(id='train-own-scatter')],style={'width': '30%', 'display': 'inline-block', 'padding': '10 10'}),
            html.P("The random forest regressor uses the following parameters : parameters = {'bootstrap': True, 'min_samples_leaf': 5, 'n_estimators': 10, 'min_samples_split': 5, 'max_features': 10, 'max_depth': 10, 'max_leaf_nodes': None}"),
            ])
            
def choose_model(model):
    if model=='RF':
        filename='final_model.pkl'
        featurelist = ['Power_kW', 'HR', 'pres_mbar', 'rain_mm/h', 'solarRad_W/m2', 'temp_C', 'windGust_m/s', 'windSpeed_m/s', 'Holiday', 'Weekday', 'Hour', 'Month', 'Power -1']
    elif model=='LR':
        filename='linear_regr.pkl'
        featurelist = ['Power_kW', 'solarRad_W/m2', 'windGust_m/s', 'Holiday', 'Weekday', 'Hour', 'Month','Power -1']
    #print(model)
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
            y=df_forecast.columns[1:])
    fig.update_layout(
            title={'text':'Comparison of Real Data and next-hour Prediction','x':0.5, 'xanchor':'center'},
            yaxis_title='Power consumption (kWh)',
            xaxis_title='Date')
    return fig

@app.callback(
        Output('scatter-ml','figure'),
        [Input('model-choose','value')])
def update_graph(model):
    df_forecast, dump, dump2 = choose_model(model)
    fig = px.scatter(df_forecast,
            x='real-data',
            y='prediction')
    fig.update_layout(
            title={'text':'Comparison in a Scatter Plot','x':0.5, 'xanchor':'center'})
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
    fig.update_layout(
            title={'text':'Exploratory Graphing of Different Features','x':0.5, 'xanchor':'center'},
            yaxis_title='Features',
            xaxis_title='Date')
    return fig

@app.callback(
    Output('model-table', 'children'),
    [Input('model-choose','value')])
def model_table(model):
    df_pred, df_metrics,dump = choose_model(model) 
    #print(df_metrics.T)
    return generate_table(df_metrics.T)
def generate_table(dataframe, max_rows=10):
    return html.Table([
        #html.Thead(
        #    html.Tr([html.Th(col) for col in dataframe.columns])
        #),
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


@app.callback(
    [Output('button-output', 'figure'),
        Output('train-own-scatter','figure'),
        Output('train-own-table','children')],
    [Input('submit-button', 'n_clicks')],
    [State('feature-checklist', 'value'),
        State('own-model-choose', 'value')])
def update_graph_cluster(button_clicks, featurelist, model):
    X_train, Y_train, t_train = split_features(featurelist)
    df_val = aug_all.copy().loc['2019']
    X = df_val[featurelist].values
    Y = df_val['Power_kW'].values
    t = df_val.index
    if model=='LR': 
        regr = linear_model.LinearRegression()
        regr.fit(X_train,Y_train)
        pred = regr.predict(X)
    elif model=='RF': 
        parameters = {'bootstrap': True,
                      'min_samples_leaf': 5,
                      'n_estimators': 10, 
                      'min_samples_split': 5,
                      'max_features': 10,#'sqrt',
                      'max_depth': 10,
                      'max_leaf_nodes': None}
        regr = RandomForestRegressor(**parameters)
        regr.fit(X_train, Y_train)
        pred = regr.predict(X)
    print(t)
    print(Y)
    d={'time':t,'real-data':Y,'prediction':pred}
    df_forecast = pd.DataFrame(data=d)
    df_metrics = ml_metrics(pred,Y)

    out = "This model uses the features: "+', '.join(featurelist)
    fig = px.line(df_forecast,
            x=df_forecast.columns[0],
            y=df_forecast.columns[1:])
    fig.update_layout(
            title={'text':'Comparison of Real Data and next-hour Prediction','x':0.5, 'xanchor':'center'},
            yaxis_title='Power consumption (kWh)',
            xaxis_title='Date')

    scatter = px.scatter(df_forecast,
            x='real-data',
            y='prediction')
    scatter.update_layout(
            title={'text':'Comparison in a Scatter Plot','x':0.5, 'xanchor':'center'},
            yaxis_title='Real Data',
            xaxis_title='Prediction')
    return fig, scatter, generate_table(df_metrics.T)




if __name__ == '__main__':
    #app.run_server(debug=True, port=8010)
    app.run_server(debug=False)
