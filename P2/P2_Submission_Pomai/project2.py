#######################################################
# Name: Pomaikai Canaday
# netid: pmc101
# homework 4
######################################################

# Import all your libraries.
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import ast
import warnings
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


######################################################
# SUMMARIZE DATA
# this method is used to summarize the data by creating
# calling describe which gives us the mean, standard 
# deviation, and quartiles. Then it prints a histogram
# and box plot for visual representation of the spread 
######################################################
def Summary(data):
	df = data
	print(df.describe())
	df.hist()
	plt.show()
	print()

	df.plot(kind='box', subplots=True, layout=(7, 8), sharex=False, sharey=False)
	plt.show()
	print()

###########################################################
# CHANGING NON-NUMERICAL DATA TO NUMERICAL DATA
# The following series of methods change the categorical
# data into non-catergorical data by converting them into 
# set values that I define. All methods following the same
# logic by creating a dictonary of the unique values in 
# a given attribute
###########################################################
def catergorical_to_numeric(dataframe):
	myData = dataframe
	myData['title'] = myData['title'].astype('category')
	myData['artist(s)'] = myData['artist(s)'].astype('category')
	myData['year'] = myData['year'].astype('category')
	categorical_data = myData.select_dtypes(['category']).columns

	myData[categorical_data] = myData[categorical_data].apply(lambda x: x.cat.codes)
	print(categorical_data)
	return dataframe

###########################################################
# CHANGING NON-NUMERICAL DATA TO NUMERICAL DATA: MANUAL FOR 
# RANK
# We did the following because we did not want random numbers
# to represent rank. Since we know that range of the rank, 
# being 1 - 100, it was not difficult to change it manually.
# this made it easier to do the binning.
###########################################################

def rankDic(dataframe):
	myData = dataframe
	rankDic = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 
	'11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20,
	'21':21, '22':22, '23':23, '24':24, '25':25, '26':26, '27':27, '28':28, '29':29, '30':30, 
	'31':31, '32':32, '33':33, '34':34, '35':35, '36':36, '37':37, '38':38, '39':39, '40':40, 
	'41':41, '42':42, '43':43, '44':44, '45':45, '46':46, '47':47, '48':48, '49':49, '50':50, 
	'51':51, '52':52, '53':53, '54':54, '55':55, '56':56, '57':57, '58':58, '59':59, '60':60, 
	'61':61, '62':62, '63':63, '64':64, '65':65, '66':66, '67':67, '68':68, '69':69, '70':70, 
	'71':71, '72':72, '73':73, '74':74, '75':75, '76':76, '77':77, '78':78, '79':79, '80':80, 
	'81':81, '82':82, '83':83, '84':84, '85':85, '86':86, '87':87, '88':88, '89':89, '90':90, 
	'91':91, '92':92, '93':93, '94':94, '95':95, '96':96, '97':97, '98':98, '99':99, '100':100}
	myData['rank'] = myData['rank'].map(rankDic)
	return myData


###########################################################
# EVALUATE ALGORITHIMS & CLASSIFICATIONS
# Using Gaussian Naives Bayes, Random Forest classifers, 
# KNeighbors Classifer, and Decision Tree Classifer. 
###########################################################
def Classifications(data, attributeAmount):
	number = attributeAmount
	the_data = data.copy()
	valueArray = the_data.values
	
	'''
	creating the 20/80 training and validation split with 
	the given data set that is passed in
	'''
	X = valueArray[:, 0:number]
	X = X.astype('int')
	Y = valueArray[:, number]
	Y = Y.astype('int')
	test_size = 0.20
	seed = 7
	X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

	######################################################
	# Use different algorithms to build models
	######################################################

	# Setup 10-fold cross validation to estimate the accuracy of different models
	# Split data into 10 parts
	# Test options and evaluation metric
	num_folds = 10
	num_instances = len(X_train)
	seed = 7
	scoring = 'accuracy'

	# Add each algorithm and its name to the model array
	model_KNN_CART = []
	model_KNN_CART.append(('KNN', KNeighborsClassifier()))
	model_KNN_CART.append(('CART', DecisionTreeClassifier()))

	# Evaluate each model, add results to a results array,
	# Print the accuracy results (remember these are averages and std
	results = []
	names = []
	for name, model in model_KNN_CART:
		kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
		cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

		model.fit(X_train, Y_train)
		model_predictions = model.predict(X_validate)

		print()
		print("Validation: " + name)
		print(accuracy_score(Y_validate, model_predictions))
		print(confusion_matrix(Y_validate, model_predictions))
		print(classification_report(Y_validate, model_predictions))


	#The following is an array of the classifiers that we use.
	models = []
	models.append(('GNB', GaussianNB()))
	models.append(('RFC', RandomForestClassifier()))

	warnings.simplefilter(action= 'ignore', category=FutureWarning) #this is to get rid of a warning I kept getting. 
	
	name = ["GNB", "RFC"] #names of the classifiers being used. This was just for print out messages. 
	
	#Iterate through the classifer array, model. Train the model using fit method
	#and then predict using x_validate
	#USED THIS LINK FOR HELP:
	#https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
	
	for name, model in models:
		with warnings.catch_warnings(): 
			warnings.filterwarnings("ignore")
			model.fit(X_train, Y_train) #training the ML classifier
			y_pred = model.predict(X_validate) #using the model to predict y values for the given X_validate set
			#print("Accuracy for ", name, ": ",metrics.accuracy_score(Y_validate, y_pred)) #Gives us the the accuracy of the classifier

			print()
			print("Validation: " + name)
			print(accuracy_score(Y_validate, model_predictions))
			print(confusion_matrix(Y_validate, model_predictions))
			print(classification_report(Y_validate, model_predictions))

###########################################################
# RUNNING HIERARCHICAL CLUSTERING FOR RANK AND ARTIST
# Using the Ward hierarchical clustering method to give us 
# more insight on the data
###########################################################
def HierarchicalClustering(myData):
	x = myData.iloc[:, [1, 2]].values
	dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
	model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
	model.fit(x)
	labels = model.labels_
	plt.scatter(x[labels==0, 0], x[labels==0, 1], s=50, marker='o', color='green')
	plt.scatter(x[labels==1, 0], x[labels==1, 1], s=50, marker='o', color='blue')
	plt.scatter(x[labels==2, 0], x[labels==2, 1], s=50, marker='o', color='red')
	plt.scatter(x[labels==3, 0], x[labels==3, 1], s=50, marker='o', color='orange')
	plt.scatter(x[labels==4, 0], x[labels==4, 1], s=50, marker='o', color='purple')
	plt.show()

	print("Silhouette Score for cluster: " + str(metrics.silhouette_score(x, labels, metric='euclidean')))

###########################################################
# BINNING DATA
# Here we are creating a new column to check the whether a 
# song is listed as a top 20 in that year. 
###########################################################
def AddingColumn(dataframe, newColumn , attribute, number):
	df = dataframe
	df[newColumn] = np.where(df[attribute] >= number, 0, 1)

	return df

###########################################################
# MAIN METHOD WHERE WE TEST EVERYTHING
###########################################################
def main():
	music_data = "music_data_new.csv"
	attributeNames = ['title', 'artist(s)', 'rank', 'year', 'genres']
	myData = pandas.read_csv(music_data, names=attributeNames)

	print(myData.mode(dropna=False))
	myData = myData.drop(myData.index[0])

	myData_cat = catergorical_to_numeric(myData)
	myData_cat = rankDic(myData_cat)

	HierarchicalClustering(myData_cat)

	Summary(myData)
	df = AddingColumn(myData_cat, 'rank >= 50', 'rank', 20)

	df = df.drop(['genres'], axis=1)

	#Hypothesis test 1: Based on the artist, title of song, and year can we predict whether the song was in the top
	# twenty of that year?
	df = df.drop(['rank'], axis=1)
	Classifications(df, 3)


if __name__ == "__main__":
	main()