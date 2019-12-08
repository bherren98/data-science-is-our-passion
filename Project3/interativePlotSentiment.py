import sys
import csv
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from ipywidgets import widgets

df = pd.read_csv('LyricData_sentiment.csv' , sep=',', encoding='latin1')

year = widgets.Dropdown(
    options=df['year'].unique().tolist(),
    value='1963',
    description='Year:',
)

# Assign an empty figure widget with traces
trace1 = go.Histogram(x=df['sentiment_score'], name='Sentiment Distribution',bins = int(30))
g = go.FigureWidget(data=[trace1],
                    layout=go.Layout(
                        title=dict(
                            text='Lyrical Sentiment Analysis, 1963 - 2018'
                        )
                    ))

def validate():
    if year.value in df['year'].unique():
        return True
    else:
        return False

def response(change):
    if validate():
        if use_date.value:
            filter_list = [i and j and k for i, j, k in
                           zip(df['year'] == year.value)]
            temp_df = df[filter_list]

        else:
            filter_list = [i and j for i, j in
                           zip(df['year'] == '1963')]
            temp_df = df[filter_list]
        x1 = temp_df['sentiment_score']
        with g.batch_update():
            g.data[0].x = x1
            g.layout.xaxis.title = 'Sentiment Scores'
            g.layout.yaxis.title = 'Number of Times'


year.observe(response, names="value")

container = widgets.HBox([year])
widgets.VBox([container,g])
