AUTHOR - Brandon Herren
DATE - 11/8/2019

Brief Use Guidelines:

Hello TA! I've provided you with a TA_DEBUG_MODE
variable in the main function that, when turned on,
runs all of my functions, including the ones that
take a long time to process CSV data files. The output
with it on is ~15 minutes. If it is turned off, the
output is only around ~30 seconds.


Files:

genreClustering.py - primary code file with... everything

/k-means output/ - all k-means cluster images (both PCA and non-PCA)

artist_data_10-06-2019_170541.json - data file from P1 with artist gender

Billboard-Year-End-Data_with_Fixed_Genres_10-07-2019_011154.json - data
file from p1 with rank/genre/title/artist info

dataCorrelations.png - heat map output of genre ratio correlations

dataHistogram.png - histogram of genre ratios

genre_correlation_11-08-2019_121532.csv - genre ratios for each initial tag

genres_11-08-2019_121248.csv - basic statistical analysis of each tag's 
release year spread/center

initial_clusters_scatter_plot_pca.png - plot of the initial classification
clusters (based on highest matching genre ratio) in PCA format

initial_clusters_scatter_plot.png - plot of the initial classification clusters
(based on highest matching genre ratio) plotted onto axes of pop ratio and
rock ratio. Includes a genre legend!

song_estimated_genres_11-08-2019_123033.csv - calculated genre weight for each
song in the original data set based on genre ratios and initial tag weights

README.txt - this file!

