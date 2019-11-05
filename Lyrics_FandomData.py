from bs4 import BeautifulSoup
from lxml import html
import requests
import json
import csv

#Json file used to look at the Artist list
filename = "Billboard-Year-End-Data_with_Fixed_Genres_10-05-2019_140320.json"
file = open(filename, 'r')
datastore = json.load(file)

temp_data = []
data = ""
allData = []

#Method for parsing through the json file to get the artist and song name
def dataCollection(jsontxt):
	#In the array of information, setting temperary variables to information specific to that zipcode
	for year in range(1963, 2019): #1963
		for song in datastore[str(year)]:
			artist= song["artist(s)"]
			music = song['song']
			#print("artist: " + str(artist) + " song: " + str(music))
			Lyrics(artist, music)

def Lyrics(artist, song):
	artist = artist
	song = song
	#url="https://lyrics.fandom.com/wiki/"+ str(artist) + ":"+ str(song)

	#try and scrape the website for the lyrics
	try:
		soup = BeautifulSoup(requests.get(url).text, 'lxml')
		#page= requests.get(url)
		#page = urllib2.urlopen(url)
		#soup = BeautifulSoup(page, 'html.parser')
		content = soup.find('div', attrs={'class': 'lyricbox'})
		#content = html_soup.find_all('div', id= 'lyricbox')
		song_lyrics = content.text.strip()
		print("Artist: " + artist + "Song: " + song + "Lyrics: " + song_lyrics)

		data = (artist, song, song_lyrics)
		#data.append(artist)
		#data.append(song)
		#data.append(song_lyrics)
		allData.append(data)
	
	#catching the exceptions if the website is not found
	except Exception as e:
		print(str(e) + "-" + "Artist: " + artist + song)

#writing to CSV method
def writetoCSV(array):
	count = 0
	csvName = "LyricData.csv"
	csvFile= open(csvName, "wt")
	playerwriter = csv.writer(csvFile, delimiter = ",")
	playerwriter.writerow(["artist", "song", "lyrics"])
	for p in range(len(allData)): 
		print("ALL DATA \n\n" + str(allData[p]))
		temp_data.append(allData[p])
	for i in range(len(temp_data)): 
		print("TEMP \n\n" + str(temp_data[i]))
		playerwriter.writerow(temp_data[i])
	csvFile.close()


dataCollection(datastore)
writetoCSV(allData)
#Lyrics("King Princess", "Pussy Is God")
#print(allData)