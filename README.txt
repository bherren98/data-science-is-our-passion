README

netID:
Pomaikai Canaday pmc101@georgetown.edu


####Notes for the TA#######
This is a lot of scraping. On estimate, it will take you 30-40 minutes for this program to run. I put in place print statements as an indicator to where you currently are in terms of going through the songs and artist. Please don't be alarmed if it takes a while. 
###########################

Billboard-Year-End-Data_with_Fixed_Genres_10-05-2019_140320.json
	- Data collected from the LastFM api call. We use the Artist and Songs from this data set to set up our call in Lyrics_FandomData.py

Lyrics_FandomData.py:
	- This contains the code to scrape the https://lyrics.fandom.com for the lyrics from the artist and songs from the Billboard. We used this website because it was easy to scrape the data. 

LyricData.csv:
	- This is the finished the data set from Lyrics_FandomData.py which contains lyrics to a specific song.
