# Author - Brandon Herren (bsh46)
# Date - 11/8/2019

import json
import csv
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from apyori import apriori
from time import strftime
import statistics


# For saving the intial json data file from P1 to CSV format
def saveJSONtoCSV(filename):
	with open(filename, encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		csv_rows = [["title", "artist(s)", "rank", "year", "genres"]]
		for year in range(1963, 2019):
			for song in billboard_chart_data[str(year)]:
				if ("genres" in song):
					csv_rows.append([song["song"], song["artist(s)"], song["rank"], year, str(song["genres"])])
				else:
					csv_rows.append([song["song"], song["artist(s)"], song["rank"], year, "[]"])


		filename = "music_hoezzz_data_" + strftime("%m-%d-%Y_%H%M%S") + ".csv"
		f = open(filename, "w", encoding="utf8", newline='')
		writer = csv.writer(f)
		writer.writerows(csv_rows)

		return filename


# use apriori analysis on genre tags used
def aprioriAnalysis(filename):
	with open(filename, encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		all_tags = []
		# get string lists of genres
		for year in range(1963, 2019):
			for song in billboard_chart_data[str(year)]:
				if ("genres" in song):
					all_tags.append(song["genres"].keys())

		# All rules of min support = 0.2, min confidence = 0.4
		print("Rules for minimum support = 0.2, minimum confidence = 0.4")
		results = list(apriori(all_tags, min_support=0.2, min_confidence=0.4))
		print(str(len(results)) + " rules found.\n")
		print(results)

		# All rules of min support = 0.3, min confidence = 0.5
		print("\nRules for minimum support = 0.3, minimum confidence = 0.5")
		results = list(apriori(all_tags, min_support=0.3, min_confidence=0.5))
		print(str(len(results)) + " rules found.\n")
		print(results)

		# All rules of min support = 0.6, min confidence = 0.6
		print("\nRules for minimum support = 0.6, minimum confidence = 0.6")
		results = list(apriori(all_tags, min_support=0.6, min_confidence=0.6))
		print(str(len(results)) + " rules found.\n")
		print(results)


# Perform basic analysis of tag values - average release year, std dev of release year,
# and a couple of other measures
def performBasicAnalysis(filename):
	with open(filename, encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		# traverse all songs
		overall_totals = {}
		genre_totals = {}
		total_weight = 0;
		# traverse all songs
		for year in range(1963, 2019):
			#print(year)
			genre_totals_year = {}
			for song in billboard_chart_data[str(year)]:
				if ("genres" in song):
					for tag in song["genres"]:
						# calculate total weights of genre tags
						if (tag in genre_totals_year):
							genre_totals_year[tag] += int(song["genres"][tag])
						else:
							genre_totals_year[tag] = int(song["genres"][tag])

						if (tag in overall_totals):
							overall_totals[tag][0] += int(song["genres"][tag])
						else:
							overall_totals[tag] = [int(song["genres"][tag]), 0, 0, []]

						overall_totals[tag][2] += 1
						overall_totals[tag][3].append(year)
						total_weight += int(song["genres"][tag])

			genre_totals[str(year)] = genre_totals_year

		# calculate number of songs each tag was used for
		for year in genre_totals:
			for tag in overall_totals:
				if overall_totals[tag][0] > 300 and tag in genre_totals[year] and genre_totals[year][tag] > overall_totals[tag][0] * 0.01:
					overall_totals[tag][1] += 1

		# write values to csv file
		csv_rows = [['name', 'total_weight', 'number_of_years', 'number_of_songs', 'release_year_mean', 'release_year_stdev']]
		for tag in overall_totals:
			if overall_totals[tag][0] > 300:
				csv_rows.append([tag, overall_totals[tag][0], overall_totals[tag][1], overall_totals[tag][2], sum(overall_totals[tag][3]) / len(overall_totals[tag][3]), statistics.stdev(overall_totals[tag][3])])

		filename = "genres_" + strftime("%m-%d-%Y_%H%M%S") + ".csv"
		f = open(filename, "w", encoding="utf8", newline='')
		writer = csv.writer(f)
		writer.writerows(csv_rows)

		return filename


#INNER CATEGORY			| BROADER CATEGORY
#-------------------------------------------
#rock	        		| contains rock
#alternative			| contains alternative
#indie					| contains indie
#pop					| contains pop
#soul					| contains soul
#dance					| contains dance
#r&b/rnb				| contains r&b/rnb
#hip hop/hip-hop/hiphop	| contains hip/hop
#rap					| contains rap
#country				| contains country
#metal					| contains metal
#jazz					| contains jazz
#electronic(a)/edm		| contains electro/edm
#folk					| contains folk

# Separates tags into genre overlap ratios based on the
# above 14 categories. Exact ratios are based on the listed
# categories, broader ratios are based on all tags that
# contain the given key words in the BROADER CATEGORY column.
def findGenreCorrelation(filename):
	with open(filename, encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		# lists out matching tags to search for
		search_terms = {
			'rock': [['rock'], ['rock']],
			'alternative': [['alternative'], ['alternative']],
			'indie': [['indie'], ['indie']],
			'pop': [['pop'], ['pop']],
			'soul': [['soul'], ['soul']],
			'dance': [['dance'], ['dance']],
			'r&b': [['r&b', 'rnb'], ['r&b', 'rnb']],
			'hip_hop': [['hip hop', 'hiphop', 'hip-hop'], ['hip', 'hop']],
			'rap': [['rap'], ['rap']],
			'country': [['country'], ['country']],
			'metal': [['metal'], ['metal']],
			'jazz': [['jazz'], ['jazz']],
			'electronic': [['electronic', 'electronica', 'edm'], ['electro', 'edm']],
			'folk': [['folk'], ['folk']]
		}

		genre_correlation = {}
		# traverse all songs
		for year in range(1963, 2019):
			print(year)
			for song in billboard_chart_data[str(year)]:
				if "genres" in song:
					for tag in song["genres"]:
						if tag not in genre_correlation:
							# set initial genre correlation values
							genre_correlation[tag] = {
								'name': tag, 
								'total_weight': 0,
								'rock': [0, 0],
								'alternative': [0, 0],
								'indie': [0, 0],
								'pop': [0, 0],
								'soul': [0, 0],
								'dance': [0, 0],
								'r&b': [0, 0],
								'hip_hop': [0, 0],
								'rap': [0, 0],
								'country': [0, 0],
								'metal': [0, 0],
								'jazz': [0, 0],
								'electronic': [0, 0],
								'folk': [0, 0]
							}

						# update genre correlation values
						genre_correlation[tag]['total_weight'] += song["genres"][tag]
						for genre_category in search_terms.keys():
							exact_overlap = 0
							broader_overlap = 0
							for tag2 in song["genres"]:
								if tag != tag2:
									# find all matching exact overlaps
									for exact_match in search_terms[genre_category][0]:
										if tag2.lower() == exact_match:
											exact_overlap += song["genres"][tag2]
											break

									# findall matching broad overlaps
									for broader_term in search_terms[genre_category][1]:
										if broader_term in tag2.lower():
											broader_overlap += song["genres"][tag2]
											break

							# set final values
							exact_overlap = min(exact_overlap, song["genres"][tag])
							broader_overlap = min(broader_overlap, song["genres"][tag])

							genre_correlation[tag][genre_category][0] += exact_overlap
							genre_correlation[tag][genre_category][1] += broader_overlap

	csv_rows = []
	# set header based on genre categories
	header = ['name', 'total_weight']
	for genre_category in search_terms.keys():
		header.append(str(genre_category + "_exact"))
		header.append(str(genre_category + "_broad"))
	csv_rows.append(header)

	# calculate final ratios and write to csv file
	for tag in genre_correlation:
		if genre_correlation[tag]['total_weight'] > 300:
			next_row = [tag, genre_correlation[tag]['total_weight']]
			for genre_category in search_terms.keys():
				# if main category, append 1
				if tag in search_terms[genre_category][0]:
					next_row.append(1)
					next_row.append(1)
				# otherwise, calculate ratio of overlap: genre weight / total weight
				else: 
					total_weight = next_row[1]
					next_row.append(genre_correlation[tag][genre_category][0] / total_weight)
					next_row.append(genre_correlation[tag][genre_category][1] / total_weight)
			csv_rows.append(next_row)

	# write to csv file
	filename = "genre_correlation_" + strftime("%m-%d-%Y_%H%M%S") + ".csv"
	f = open(filename, "w", encoding="utf8", newline='')
	writer = csv.writer(f)
	writer.writerows(csv_rows)

	return filename


# Uses tag genre overlap values to assign genre weights to each song in the
# original data set. Returns CSV file with the following columns:
# [year, rank, title, artist, closest match, genre0, genre1, ...]
def getEstimatedGenreOverlapBySong(filename, data_frame):
	data_frame = data_frame.drop(columns=['total_weight'])

	with open(filename, encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		csv_header = ['year', 'rank', 'title', 'artist', 'closest match']
		for genre_category in data_frame.columns.values[1:]:
			csv_header.append(genre_category)
		csv_rows = [csv_header]
		# traverse all songs
		for year in range(1963, 2019):
			print(year)
			for song in billboard_chart_data[str(year)]:
				print(song['rank'])
				# initial next song rows
				next_row = [year, song['rank'], song['song'], song['artist(s)'], '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				if ("genres" in song):
					for tag in song["genres"]:
						if tag in data_frame["name"].values:
							# calculate genre weights based on initial tag weights and tag overlap ratios
							for i, genre_category in enumerate(data_frame.columns.values[1:]):
								next_row[5 + i] += song["genres"][tag] * data_frame.loc[data_frame.index[data_frame['name'] == tag].tolist()[0], genre_category]

				# add highest weight genre to the data frame
				max_value = max(next_row[5:])
				if (max_value == 0):
					next_row[4] = "Unknown"

				elif (max_value < 0.25):
					next_row[4] = "Insufficient"

				else:
					next_row[4] = data_frame.columns.values[1:][next_row[5:].index(max_value)]

				csv_rows.append(next_row)

		# write to csv
		filename = "song_estimated_genres_" + strftime("%m-%d-%Y_%H%M%S") + ".csv"
		f = open(filename, "w", encoding="utf8", newline='')
		writer = csv.writer(f)
		writer.writerows(csv_rows)

		return filename


#Runs DBScan analysis for a set eps on a given data set
def runDBScan(eps, data_frame):
	#Pre-processing
	x = data_frame.values # returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	normalizedDataFrame = pd.DataFrame(x_scaled)

	dbscan = DBSCAN(eps=eps)
	# if no eps value was specified
	if (eps == -1):
		dbscan = DBSCAN()
	cluster_labels = dbscan.fit_predict(normalizedDataFrame)

	# n_clusters/n_noise Output taken from Scikit example
	# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py	
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
	n_noise_ = list(dbscan.labels_).count(-1)

	print('\nEvaluation of DBSCAN')
	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)
	
	# Determine if the clustering is good
	silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
	print("For eps =", eps, "The average silhouette_score is :", silhouette_avg)


#Runs k-means analysis for a set number of clusters on a given data set
def runKMeans(num_clusters, data_frame):
	#Determine size of plots
	s = [pow(w, 0.5) for w in data_frame['total_weight']]
	data_frame = data_frame.drop(columns=['total_weight'])
	#initial_genres = data_frame.idxmax(axis=1)
	initial_genres = np.argmax(data_frame.values, axis=1)

	#Pre-processing
	x = data_frame.values # returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	normalizedDataFrame = pd.DataFrame(x_scaled)

	kmeans = KMeans(n_clusters=num_clusters)
	cluster_labels = kmeans.fit_predict(normalizedDataFrame)

	# Determine if the clustering is good
	silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
	print("For n_clusters =", num_clusters, "The average silhouette_score is :", silhouette_avg)

	#Print arrays to console
	centroids = kmeans.cluster_centers_
	pprint(cluster_labels)
	pprint(centroids)

	# Plot clusters (with fun colors!)
	plt.style.use = 'default'
	pl.suptitle("Scatter Plot for K-means of n=" + str(num_clusters))
	plt.scatter(x[:, 0], x[:, 1], c=cluster_labels, s=s, cmap='plasma')
	plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.5);
	plt.savefig("k_means_n_" + str(num_clusters) + "_" + strftime("%m-%d-%Y_%H%M%S"))
	plt.clf()

	# Plot initial genres (with fun colors!)
	plt.style.use = 'default'
	plt.ion()
	pl.suptitle("Scatter Plot for Initial Genre Inputs")
	plt.scatter(x[:, 0], x[:, 1], c=initial_genres, s=s, cmap='hsv')
	#plt.show(block=True)
	plt.savefig("initial_clusters_scatter_plot")
	plt.clf()

	# Plot clusters using PCA analysis
	pca = PCA(n_components= 14)
	pca.fit(x)
	x = pca.transform(x)

	plt.style.use = 'default'
	pl.suptitle("Scatter Plot (PCA) for K-means of n=" + str(num_clusters))
	plt.scatter(x[:, 0], x[:, 1], c=cluster_labels, s=s, cmap='plasma')
	plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.5);
	plt.savefig("k_means_n_pca_" + str(num_clusters) + "_" + strftime("%m-%d-%Y_%H%M%S"))
	plt.clf()

	# Plot initial genres with PCA (but still fun colors!)
	plt.style.use = 'default'
	plt.ion()
	pl.suptitle("Scatter Plot for Initial Genre Input (with PCA)")
	plt.scatter(x[:, 0], x[:, 1], c=initial_genres, s=s, cmap='hsv')
	#plt.show(block=True)
	plt.savefig("initial_clusters_scatter_plot_pca")
	plt.clf()


#Outputs histograms for the genre overlap values
def createHistogramsByGenreCategory(data_frame):
	# Basic plotting - histogram of entire dataframe
	# Visualize basic stats & print plot to a file
	data_frame.hist()
	plt.title("Distribution of different variables")
	# plt.show()
	plt.savefig('dataHistogram.png')
	plt.clf()


# Find correlations of genre categories and plot them to file
def plotGenreCorrelations(data_frame):
	plt.matshow(data_frame.corr())
	print(data_frame.corr().to_string() + "\n")
	plt.title("Correlation of Genre Categories")
	# plt.show()
	plt.savefig('dataCorrelations.png')
	plt.clf()


# runs classification on training/test data for given model
# prints out accuracy output
def classify(data_frame, model_name, model):
	valueArray = data_frame.values
	X = valueArray[:, 0:-1]
	Y = valueArray[:, -1]
	Y = Y.astype('int')
	test_size = 0.20
	seed = 7
	X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

	num_folds = 10
	num_instances = len(X_train)
	seed = 7
	scoring = 'accuracy'

	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	msg = "%s: %f (%f)" % (model_name, cv_results.mean(), cv_results.std())
	print(msg)

	model.fit(X_train, Y_train)
	model_predictions = model.predict(X_validate)

	print()
	print(accuracy_score(Y_validate, model_predictions))
	print(confusion_matrix(Y_validate, model_predictions))
	print(classification_report(Y_validate, model_predictions))


# Normalizes numeric/categorical variables in data frame
# prior to classification.
def getProcessedDataFrameForClassification(filename):
	myData = pd.read_csv(filename)
	myData = myData[myData['closest match'] != 'Unknown']
	myData['class'] = np.where(myData['year'] < 1990, 'True', 'False')
	myData = myData.drop(columns=['rank', 'year'])

	x = myData.values
	x_nums = x[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] # all numeric variables
	# normalize all numeric variables
	x_normalized = preprocessing.normalize(x_nums)
	x_categorical = x[:, [0, 1, 2, 17]] # all categorical variables

	count = 0
	categoricalAttributes = ['title', 'artist', 'closest match', 'class']
	for attribute in categoricalAttributes:
		uniqueValues = pd.unique(myData[attribute])
		labelEncoder = preprocessing.LabelEncoder()
		labelEncoder.fit(uniqueValues)
		data_column = labelEncoder.transform(myData[attribute])
		x_categorical [:,count] = np.transpose(data_column)

		count += 1

	x_normalized = np.concatenate((x_normalized, x_categorical), axis=1)
	attributeNames = ['rock', 'pop', 'indie', 
						'alternative', 'soul', 'r&b', 'dance', 
						'hip_hop', 'rap', 'country', 'metal', 
						'jazz', 'electronic', 'folk', 
						'title', 'artist', 'closest match', 'class']
						
	normalizedDataFrame = pd.DataFrame(x_normalized, columns=attributeNames)
	print(normalizedDataFrame[:10].to_string())

	return normalizedDataFrame


def main():
	# keep this on if you want to see all of my csv file things run and make sure they're right
	# turn it off if you're re-running the file and you don't want to wait 15 minutes for output
	TA_DEBUG_MODE = False 

	# Project 1 Data File
	filename_initial = "Billboard-Year-End-Data_with_Fixed_Genres_10-07-2019_011154.json"

	# Perform apriori analysis on list of Genre tags
	aprioriAnalysis(filename_initial)

	# Run Analysis to get Std Dev, Mean, etc for each genre tag
	if TA_DEBUG_MODE:
		filename = performBasicAnalysis(filename_initial)

	# Selected genre categories
	genres = ['rock', 'pop', 'indie', 'alternative', 'soul', 'r&b',
				'dance', 'hip_hop', 'rap', 'country', 'metal', 'jazz',
				'electronic', 'folk']

	# Find overlap of genre tags with 14 categories
	filename2 = "genre_correlation_11-08-2019_121532.csv"
	if TA_DEBUG_MODE:
		filename2 = findGenreCorrelation(filename_initial)
	myData2 = pd.read_csv(filename2)

	# Concatenate necessary rows
	rows_to_concat = [myData2["name"], myData2["total_weight"]]
	for genre in genres:
		rows_to_concat.append(myData2[genre + "_exact"])
	myData2 = pd.concat(rows_to_concat, axis=1, keys=["name", "total_weight"] + genres)

	# Calculate genre weights for individual songs based on filename2
	filename3 = "song_estimated_genres_11-08-2019_123033.csv"
	if TA_DEBUG_MODE:
		filename3 = getEstimatedGenreOverlapBySong(filename_initial, myData2)

	# Perform pre-processing before classification
	normalizedDataFrame = getProcessedDataFrameForClassification(filename3)

	# Gaussian Naive Bayes
	classify(normalizedDataFrame, "GaussianNB", GaussianNB())

	# Random Forest Classifier
	classify(normalizedDataFrame, "Random Forest", RandomForestClassifier(n_estimators=100))

	# Drop name column before further analysis
	myData2 = myData2.drop(columns=['name'])

	# Perform histogram analysis for genre overlap values
	createHistogramsByGenreCategory(myData2)

	# Get correlation of overlap values
	plotGenreCorrelations(myData2)

	# Run K-Means / DBSCAN clustering methods on each tag's overlap values
	runKMeans(2, myData2)
	runKMeans(4, myData2)
	runKMeans(14, myData2)
	runDBScan(.65, myData2)


#Calls main() function
if __name__ == '__main__':
	main()