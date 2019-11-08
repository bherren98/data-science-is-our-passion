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
	myData['song'] = myData['song'].astype('category')
	myData['artist(s)'] = myData['artist(s)'].astype('category')
	myData['year'] = myData['year'].astype('category')
	categorical_data = myData.select_dtypes(['category']).columns

	myData[categorical_data] = myData[categorical_data].apply(lambda x: x.cat.codes)
	print(categorical_data)
	return dataframe

###########################################################
# CHANGING NON-NUMERICAL DATA TO NUMERICAL DATA: MANUAL FOR 
# RANK AND GENDER
# We did the following because we did not want random numbers
# to represent rank and gender. Since we know that range of 
# the rank, being 1 - 100 and we know the categories for 
# gender (female, male, and unknown (bands/multiple artist)), 
# it was not difficult to change it manually.
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

def genderDic(dataframe):
	myData = dataframe
	gender = {'female': 0, 'male':1, 'unknown':3}
	myData['gender'] = myData['gender'].map(gender)
	return myData


###########################################################
# EVALUATE ALGORITHIMS & CLASSIFICATIONS
# Using Gaussian Naives Bayes and Random Forest classifers
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

			model.fit(X_train, Y_train)
			model_predictions = model.predict(X_validate)

			print()
			print("Validation: " + name)
			print(accuracy_score(Y_validate, model_predictions))
			print(confusion_matrix(Y_validate, model_predictions))
			print(classification_report(Y_validate, model_predictions))


###########################################################
# BINNING DATA
# Here we are creatin
###########################################################
def AddingColumn(dataframe, newColumn , attribute, number):
	df = dataframe
	df[newColumn] = np.where(df[attribute] == number, 0, 1)
	return df

###########################################################
# MAIN METHOD WHERE WE TEST EVERYTHING
###########################################################
def main():

	gender_data = "artistGenderRankData.csv"
	attributeNames = ['artist(s)', 'rank', 'song', 'year', 'gender']
	myData = pandas.read_csv(gender_data, names=attributeNames)

	#print(myData.dtypes)
	print(myData.mode(dropna=False))

	myData_cat = catergorical_to_numeric(myData)
	myData_cat = rankDic(myData_cat)
	myData_cat = genderDic(myData_cat)
	Summary(myData_cat)
	df = AddingColumn(myData_cat, 'Band/Multiple Artist', 'gender', 3)

	df = df.drop(['gender'], axis=1)

	#Hypothesis test 1: Based on the artist, rank, and year can we predict whether the artist was a band/mulitple artist?
	Classifications(df, 4)


if __name__ == "__main__":
	main()