import pylast
from time import strftime
import xlrd, xlwt, csv
import json

API_KEY = "0dcacf9308d8c7cb06d900e746ab5cf9"
API_SECRET = "66781de7c26ce6ab32b9bdcc14d550cf"
username = "bherren98"

password_hash = pylast.md5("DataIsDifficult1!")

network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET, username=username, password_hash=password_hash)


# Runs API calls on each song in the json
def getInitialGenreData(filename):
	with open(filename, encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		count = 1
		# traverse each song in the json
		for year in range(1963, 2019):
			for song in billboard_chart_data[str(year)]:
				song_name = song['song']
				artist_name = song['artist(s)']

				try:
					# use last.fm api (pylast) to find corresponding track and genre tags
					track = network.get_track(artist_name, song_name)
					tags = track.get_top_tags()
					mbid_id = track.get_mbid() # used for musicbrainz api in other script

					saved_tags = {}
					print(str(count) + ". " + artist_name + ": " + song_name)
					count = count + 1

					# add all tags that have a weight of at least 5 (have 5% as many tags as the top tag for the song)
					for tag in tags:
						if (int(tag.weight) >= 5):
							saved_tags[tag.item.name] = int(tag.weight)

					# save in the dict
					song["genres"] = saved_tags
					song["mbid_artist_id"] = mbid_id

				except Exception as e:
					#error = True
					print(str(e) + " - " + song_name)

		# write to new json file
		output_filename = "Billboard-Year-End-Data_with_Genre_" + strftime("%m-%d-%Y_%H%M%S") + ".json"
		with open(output_filename, "w+") as output_file:
			json.dump(billboard_chart_data, output_file)

		return output_filename

# Helper method for getting genres, used in data fixing
def getGenres(artist_name, song_name):
	saved_tags = {}
	track = 0
	mbid_id = 0

	try:
		track = network.get_track(artist_name, song_name)
		tags = track.get_top_tags() # genre tags
		mbid_id = track.get_mbid() # used in later script

		for tag in tags:
			if (int(tag.weight) >= 5):
				saved_tags[tag.item.name] = int(tag.weight)
		
	except Exception as e:
		#error = True
		print(str(e) + " - " + song_name)

	return [saved_tags, mbid_id]

# STEPS FOR FIXING MISSING DATA                                                    | WORKS FOR:           | CUMULATIVE:   |
# ---------------------------------------------------------------------------------|--------------------------------------|
# 0. run basic api calls on all songs                                              | 5488 / 5900 (93.0%)  | 5488 / 5900   |
# 1. try with split song by "/" for a-side/b-sides                                 |   18 / 5900 (00.3%)  | 5506 / 5900   |
# 2. try with split artist by "featuring"                                          |  256 / 5900 (04.3%)  | 5762 / 5900   |
# 3. try with split artist by "featuring" and move "feat." + other artist to song  |   15 / 5900 (00.3%)  | 5777 / 5900   |
# 4. try with split artist by ","                                                  |    7 / 5900 (00.1%)  | 5784 / 5900   |
# 5. try with split artist by "and"                                                |    8 / 5900 (00.1%)  | 5792 / 5900   |
# 6. try replacing "and" with "&"                                                  |   22 / 5900 (00.4%)  | 5814 / 5900   | 
# 7. try removing parts of the song in parenthesis                                 |    8 / 5900 (00.1%)  | 5822 / 5900   |
# INCOMPLETE ROWS - Genre data could not be found and is missing in JSON           |   78 / 5900 (01.3%)  | 5900 / 5900   |

# This function makes fixes to song names and artist names that couldn't be found in their basic form by the API. Steps
# addressed above. Cleans data!
def fixMissingGenreData(filename):
	with open(filename, encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		count = 1
		# traverse all songs
		for year in range(1963, 2019):
			for song in billboard_chart_data[str(year)]:
				genres_fixed = False
				song_name = song['song']
				artist_name = song['artist(s)']

				# for all songs with missing genre data
				if (("genres" not in song or len(song['genres']) == 0)):
					#print(str(year) + ": " + artist_name + " - " + song_name)
					
					# 1. try with split song by "/" for a-side/b-sides                                |
					if ("/" in song_name):
						song_name_fix_1 = song_name.split("/")[0].strip()
						if (song_name_fix_1[-1:] == '"'):
							song_name_fix_1 = song_name_fix_1[:-1]

						results = getGenres(artist_name, song_name_fix_1)
						genres = results[0]
						song['mbid_artist_id'] = results[1]
						if (len(genres) != 0):
							song["genres"] = genres
							genres_fixed = True
							print(str(count) + ": " + song_name + " - " + artist_name)
							count += 1
					
					# 2. try with split artist by "featuring"
					if ("featuring" in artist_name and not genres_fixed):
						artist_name_fix_2 = artist_name.split("featuring")[0].strip()

						results = getGenres(artist_name_fix_2, song_name)
						genres = results[0]
						song['mbid_artist_id'] = results[1]
						if (len(genres) != 0):
							song["genres"] = genres
							genres_fixed = True
							print(str(count) + ": " + song_name + " - " + artist_name)
							count += 1
					
					# 3. try with split artist by "featuring" and move "feat." + other artist to song
					if ("featuring" in artist_name and not genres_fixed):
						artist_name_fix_3 = artist_name.split(" featuring")[0]
						song_name_fix_3 = song_name + " (feat. " + artist_name.split(" featuring ")[1] + ")"
						
						results = getGenres(artist_name_fix_3, song_name_fix_3)
						genres = results[0]
						song['mbid_artist_id'] = results[1]
						if (len(genres) != 0):
							song["genres"] = genres
							genres_fixed = True
							print(str(count) + ": " + song_name + " - " + artist_name)
							count += 1
					
					# 4. try with split artist by ","
					if ("," in artist_name and not genres_fixed):
						artist_name_fix_4 = artist_name.split(",")[0]

						results = getGenres(artist_name_fix_4, song_name)
						genres = results[0]
						song['mbid_artist_id'] = results[1]
						if (len(genres) != 0):
							song["genres"] = genres
							genres_fixed = True
							print(str(count) + ": " + song_name + " - " + artist_name)
							count += 1
					
					# 5. try with split artist by "and"
					if ("and" in artist_name and not genres_fixed):
						artist_name_fix_5 = artist_name.split("and")[0]

						results = getGenres(artist_name_fix_5, song_name)
						genres = results[0]
						song['mbid_artist_id'] = results[1]
						if (len(genres) != 0):
							song["genres"] = genres
							genres_fixed = True
							print(str(count) + ": " + song_name + " - " + artist_name)
							count += 1
					
					# 6. try replacing "and" with "&"
					if ("and" in artist_name and not genres_fixed):
						artist_name_fix_6 = artist_name.replace("and", "&")

						results = getGenres(artist_name_fix_6, song_name)
						genres = results[0]
						song['mbid_artist_id'] = results[1]
						if (len(genres) != 0):
							song["genres"] = genres
							genres_fixed = True
							print(str(count) + ": " + song_name + " - " + artist_name)
							count += 1

					# 7. try removing parts of the song in parenthesis
					if ("(" in song_name and ")" in song_name and not genres_fixed):
						open_parenthesis_location = song_name.index("(")
						closed_parenthesis_location = song_name.index(")")

						substring_to_remove = song_name[open_parenthesis_location:closed_parenthesis_location+1]
						song_name_fix_7 = song_name.replace(substring_to_remove, "").strip()
						results = getGenres(artist_name, song_name_fix_7)
						genres = results[0]
						song['mbid_artist_id'] = results[1]
						if (len(genres) != 0):
							song["genres"] = genres
							genres_fixed = True
							print(str(count) + ": " + song_name + " - " + artist_name)
							count += 1

					# print remaining songs unfixed
					if (not genres_fixed):
						print(str(year) + ": " + artist_name + " - " + song_name)

		# save updated json file if any songs were fixed
		if (count > 1):
			with open("Billboard-Year-End-Data_with_Fixed_Genres_" + strftime("%m-%d-%Y_%H%M%S") + ".json", "w+") as output_file:
				json.dump(billboard_chart_data, output_file)

# NOT FULLY IMPLEMENTED - will be used for genre clustering

"""def analyzeGenreData():
	with open("Billboard-Year-End-Data_with_Fixed_Genres_10-05-2019_140320.json", encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		genre_totals = {}
		for year in range(1963, 2019):
			genre_totals_year = {}
			for song in billboard_chart_data[str(year)]:
				if ("genres" in song):
					for tag in song["genres"]:
						if (tag in genre_totals_year):
							genre_totals_year[tag] += int(song["genres"][tag])
						else:
							genre_totals_year[tag] = int(song["genres"][tag])
			genre_totals[str(year)] = genre_totals_year"""

def main(): 
	# make initial api calls
	#input_filename = "Billboard-Year-End-Data_10-03-2019_113905.json"
	#output_filename = getInitialGenreData(input_filename)

	# clean the data!
	output_filename = "Billboard-Year-End-Data_with_Genre_10-07-2019_005844.json"
	fixMissingGenreData(output_filename)

	# analyze the data!
	#analyzeGenreData()


#Calls main() function
if __name__ == '__main__':
	main()



