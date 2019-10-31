from bs4 import BeautifulSoup
import requests
import xlrd, xlwt, csv
from time import strftime
import json


# is the input a valid integer
def isAnInt(rank):
    try: 
        number = int(rank)
        return True
    except ValueError:
        return False

# scrapes top 100 songs from wikipedia 
def scrapeBillboardData():
	chart_data = {}

	for year in range(1963, 2019):
		# request url
		result = requests.get("https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_" + str(year))

		soup = BeautifulSoup(result.content)

		# use beautiful soup to search through rows
		song_table_rows = soup.find_all("tr")
		chart_data[year] = []
		for i in range(len(song_table_rows)):
			rank = -1
			song_name = ""
			song_artist = ""

			table_entries = song_table_rows[i].find_all("td")
			table_headers = song_table_rows[i].find_all("th")

			# formatting is slightly different for years post-1982
			if (year >= 1982):
				if (len(table_headers) > 0):
					rank = table_headers[0].text
				if (len(table_entries) > 1):
					song_name = table_entries[0].text[1:][:-1]
					song_artist = table_entries[1].text
			else:
				if (len(table_entries) > 2):
					rank = table_entries[0].text
					song_name = table_entries[1].text[1:][:-1]
					song_artist = table_entries[2].text
			
			# if num is valid, save to dict
			if (isAnInt(rank) and int(rank) > 0):
				print(str(int(rank)) + ": " + song_name + " by " + song_artist)

				song_artist = song_artist[:-1]
				
				chart_data[year].append({
					'rank': int(rank),
					'song': song_name,
					'artist(s)': song_artist 
				})

	# dump to json
	with open("Billboard-Year-End-Data_" + strftime("%m-%d-%Y_%H%M%S") + ".json", "w+") as output_file:
		json.dump(chart_data, output_file) 

def main():
	scrapeBillboardData()

#Calls main() function
if __name__ == '__main__':
	main()



