#######################################
## Assignment: Project 3, Sentiment Analysis
#######################################
import sys
import csv
import numpy as np
import pandas as pd
from pprint import pprint         
from textblob import TextBlob 
from itertools import islice
import nltk
import plotly
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns

##############################################################################
## This is where we will be doing the sentiment analysis
## Used the following website for reference
## https://towardsdatascience.com/statistical-sentiment-analysis-for-survey-data-using-python-9c824ef0c9b0
##############################################################################
def SentimentAnalysis(dataframe, newDataFrame, COLS):
	df_lyricData = dataframe
	myDataFrame = newDataFrame

	for index, row in islice(df_lyricData.iterrows(), 0, None):
		new_entry = []
		text=row['lyrics']
		blob = TextBlob(text)

		sentiment_score = blob.sentiment.polarity
		if sentiment_score == 0:
			sentiment = 'neutral'
		elif sentiment_score > 0:
			sentiment = 'positive'
		else:
			sentiment = 'negative'

		new_entry += [row['year'],row['artist'],row['song'],text,sentiment_score, sentiment]
		single_LyricSentimet_df = pd.DataFrame([new_entry], columns=COLS)
		myDataFrame = myDataFrame.append(single_LyricSentimet_df, ignore_index=True)
	return myDataFrame

#creating the new CSV file with the Sentiment Analysis
def toCSV(myDataFrame,filename, COLS):
	myDataFrame = myDataFrame
	filename = filename 
	myDataFrame.to_csv(filename, mode='w', columns=COLS, index=False, encoding="utf-8")

#Sentiment Score Distribution for dataFrame
def Sentiment_Distribution(dataframe):
	myDataFrame = dataframe
	plt.hist(myDataFrame['sentiment_score'], color = 'darkred', edgecolor = 'black', density=False,
         bins = int(30))
	plt.title('Sentiment Score Distribution')
	plt.xlabel("Sentiment Score")
	plt.ylabel("Number of Times")
	plt.show()

#This is to get a Categorical Barplot
def CategoricalBarPlot(dataframe, attribute):
	myDataFrame = dataframe
	attribute = attribute
	sns.catplot(x=attribute, kind="count", palette="ch:.25", data=myDataFrame)
	plt.show()

def main():
	COLS = ['year', 'artist','song', 'lyrics', 'sentiment_score', 'sentiment']
	myDataFrame = pd.DataFrame(columns=COLS)
	df_lyricData = pd.read_csv('LyricData.csv' , sep=',', encoding='latin1')

	myDataFrame = SentimentAnalysis(df_lyricData, myDataFrame, COLS)
	toCSV(myDataFrame,'LyricData_sentiment.csv', COLS)
	Sentiment_Distribution(myDataFrame)
	CategoricalBarPlot(myDataFrame, 'sentiment')


if __name__ == "__main__":
	main()

