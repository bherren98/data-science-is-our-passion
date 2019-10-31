Author: Brandon Herren (bsh46)
Date: 10/7/2019

Files:

billboard_wikipedia_scraping.py:
	Scrapes Billboard Year-End Top 100 data from 1963 to 2018.

last_fm_genre_finder.py
	Scrapes genre tags from last.fm API for each top song.

musicbrainz_gender_determiner.py
	Scrapes gender/group status from musicbrainz API for each top song.

Billboard-Year-End-Data_DATETIME.json
	Output file that includes rank, song name, artist name, year

Billboard-Year-End-Data_with_Genre_DATETIME.json
	Output file that includes rank, song name, artist name, year, genre tags (uncleaned) | 412 incomplete columns

Billboard-Year-End-Data_with_Fixed_Genres_DATETIME.json
	Output file that includes rank, song name, artist name, year, genre tags (cleaned) | 78 incomplete columns

### COULD NOT GET BECAUSE API WAS DOWN SO KEPT ORIGINAL VERSION OF DATA
Billboard-Year-End-Data_with_Fixed_Genres_and_Gender_DATETIME.json
	Output file that includes rank, song name, artist name, year, genre tags (cleaned), type, gender

### ORIGINAL VERSION OF DATA
artist_data_DATETIME.json
	Output file that includes artist name, type, gender | 401 incomplete columns

README.txt
	this file!