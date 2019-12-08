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
df = pd.read_csv('artistGenderRankData.csv' , sep=',', encoding='latin1')

print(df[:10])

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider', figure={
            'data': [
                {
                    'x': df['gender'],
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
    genderList = []
    for index, row in df.iterrows():
            if row['year'] == selected_year:
                genderList.append(row['gender'])
    femaleCount = genderList.count('female')
    maleCount = genderList.count('male')
    unknownCount = genderList.count('unknown')

    labels = ['female', 'male', 'unknown']
    values = [femaleCount, maleCount, unknownCount]
    return {
        "data": [go.Pie(labels=labels, values= values,
                        marker={'colors': ['#EF963B', '#C93277', '#349600', '#EF533B', '#57D4F1']}, textinfo='label')],
        "layout": go.Layout(title=f"Gender Yearly", margin={"l": 300, "r": 300, },
                            legend={"x": 1, "y": 0.7})}


if __name__ == '__main__':
    app.run_server(debug=True)
    app.server.run()
