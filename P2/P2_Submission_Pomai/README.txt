########################################
project2.py and project2_gender.py
README TEXT FILE
########################################

Artist_Rank_HCluster.jpg
	- This is a visual for the Hierarchical Clustering done in project2.py script

artistGenderRankData.csv
	- This is the data file that is used in project2_gender.py, this includes the following attributes:
		artist(s), rank, song, year, and gender

music_data.csv
	- This is the data file that was used in project2.py, this includes the following attributes:
		 title, artist(s), rank,  year, and genres
	- However we found out that there were multiples certain columns so Brandon cleaned the data set and 
	was given the following:

music_data_new.csv
	- This is the cleaned version of music_data.csv as mentioned about. This is used in project2.py

project2.py
	- This is the script that will run the following:
		- Descriptive analysis on all attributes in music_data_new.csv
		- Hierarchical Clustering with Silhoutte Score
		- Hypothesis test on the following:
			- Hypothesis test: Based on the artist, title of song, and year can we predict whether the song was in the top twenty of that year?
			- Used all classified classifers on the project 2 guideline sheet. So we used the following: A decision tree, A lazy learner, Naive Bayes, and Random Forest. 

project2_gender.py
	- This is the script that will run the following:
		- Descriptive analysis on all attributes in artistGenderRankData.csv
	- Hypothesis test on the following:
		- Hypothesis test: Based on the artist, song, rank, and year can we predict whether the artist was a band/multiple artist?
		- Used all classified classifers on the project 2 guideline sheet. So we used the following: A decision tree, A lazy learner, Naive Bayes, and Random Forest. 

