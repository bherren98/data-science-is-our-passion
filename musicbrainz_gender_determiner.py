from time import strftime
import urllib.request
import json
import musicbrainzngs

musicbrainzngs.auth("bherren", "DataIsDifficult")
musicbrainzngs.set_hostname("test.musicbrainz.org")

musicbrainzngs.set_useragent("Project to determine artist gender", "0.1", "bsh46@georgetown.edu")

# run musicbrainz api to get gender and type for each artist
def assignGenderAndTypeToSongs(filename):
	with open(filename, encoding="utf-8") as json_file:
		billboard_chart_data = json.load(json_file)

		# traverse all songs
		count = 1
		for year in range(1963, 2019):
			for song in billboard_chart_data[str(year)]:
				genres_fixed = False
				song_name = song['song']
				artist_name = song['artist(s)']
				mbid = song['mbid_artist_id']

				try:
					mb_artist = musicbrainzngs.get_artist_by_id(mbid)

					# get gender if it exists
					if ("gender" in mb_artist["artist"]):
						song['artist_gender'] = mb_artist['artist']['gender']
						print(artist_name + " is a " + song['artist_gender'])
					else:
						song['artist_gender'] = "unknown" # assigned value for when cannot be found

					# get type if it exists
					if ("type" in mb_artist["artist"]):
						song['artist_type'] = mb_artist['artist']['type']
						print(artist_name + " is a " + song['artist_type'])
					else:
						song['artist_type'] = "unknown" # assigned value for when cannot be found
				except Exception as e:
					print(str(e))

		# save to json output
		output_filename = "Billboard-Year-End-Data_with_Fixed_Genres_and_Gender" + strftime("%m-%d-%Y_%H%M%S") + ".json"
		with open(output_filename, "w+") as output_file:
			json.dump(billboard_chart_data, output_file)

		return output_filename

def main():
	filename = "Billboard-Year-End-Data_with_Fixed_Genres_10-07-2019_011154.json"
	assignGenderAndTypeToSongs(filename)

#Calls main() function
if __name__ == '__main__':
	main()