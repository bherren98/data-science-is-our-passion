import dash
import dash_core_components as dcc
import dash_html_components as html
import sys
import csv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from flask import Flask

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask('my app')


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)
df = pd.read_csv('LyricData_sentiment.csv' , sep=',', encoding='latin1')

print(df[:10])

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider', figure={
            'data': [
                {
                    'x': df['sentiment_score'],
                    'type': 'histogram'
                },
            ],
            'layout': {}
        }),
    dcc.Slider(
        id='year-slider',
        min=df['year'].min(),
        max=df['year'].max(),
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        step=None
    )
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('year-slider', 'value')])
def update_figure(selected_year):
    dff = df[df.year == selected_year]

    trace = go.Histogram(
        x = dff['sentiment_score'],
        marker = {'colorscale': 'Viridis'},
        name ='Open'
    )

    data = [trace]

    layout ={ 'title':'Lyrical Sentiment Distribution from 1963-2018', 'xaxis':{'title': 'Sentiment Distribution of Lyrics','range':[-1, 1]},
            'yaxis':{'title': 'Number of Times'}}

    fig_histogram = dict(data = data, layout = layout)
    return fig_histogram


if __name__ == '__main__':
    app.run_server(debug=True)
    app.server.run()