#Project 1: for cleaning the data for billboard
import json


def main():
    
    with open('BillboardData.json') as json_file: #opens the billboard.json file
        myData = json.load(json_file) #loads data into a json file
        
        getMissing(myData) #call to get the missing data of the file

def getMissing(myData):
    rankcount = 0 #count for ranks that are missing
    songcount = 0 #count for songs that are missing
    artistcount = 0 #count for artists that are missing
    genrecount = 0 #count for genres that are missing
    noisevalues = 0 #count for noise values in data set
    pcount = 0 #count of rankings/songs
    totalcount = 0 #totalcount of all the attributes in the file

    for year in range(1963, 2019): #for each year present in the file
        for p in myData[str(year)]: #for each song/rank in that year
            #print(str(p) + "\n")
            pcount = pcount + 1 #add a count of the ranking/songs (a "row")
            totalcount = totalcount + 4 #add the totalcount of attributes (+4 more)
            if not p.get("rank"): #check if there is not a value for rank
                rankcount = rankcount + 1
            if not p.get("song"): #check if there is not a value for song
                songcount = songcount + 1
            if not p.get("artist(s)"): #check if there is not a value for artists
                artistcount = artistcount + 1
            if not p.get("genres"): #check if there is not a value for genre
                genrecount = genrecount + 1
            
            #checking for noise values
            if type(p['rank']) != int: #if rank is not of type int
                noisevalues = noisevalues + 1
            if type(p['song']) != str: #if song is not of type string
                noisevalues = noisevalues + 1
            if type(p['artist(s)']) != str: #if artist is not of type string
                noisevalues = noisevalues + 1
            if p.get("genres"): #making sure there is a mapping to a genres value
                if not isinstance(p['genres'], dict): #if genres is not a dict
                    noisevalues = noisevalues + 1
            
    #printing out corresponding values
    print("The fraction of missing values for ranks is: " + str(rankcount/pcount) + "\n")
    print("The fraction of missing values for songs is: " + str(songcount/pcount) + "\n")
    print("The fraction of missing values for artists is: " + str(artistcount/pcount) + "\n")
    print("The fraction of missing values for genres is: " + str(genrecount/pcount) + "\n")
    
    print("The fraction of noise values is : " + str(noisevalues/totalcount) + "\n")
#getMissing
    
    
main()
