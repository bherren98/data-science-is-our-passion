from pandas.io.json import json_normalize
import pandas as pd
import json

def main():
    myData = readData()
    #descriptiveAnalytics(myData)
    
    
def readData():
    artistGenderData = pd.read_json('artist_data_10-06-2019_170541.json')
    artistGenderData.to_csv("artistGenderData.csv")
    artistSongRank = pd.DataFrame()
    
    with open('Billboard-Year-End-Data_10-03-2019_113905.json') as dataFile:
        data = json.load(dataFile)
    
    for year in range(1963, 2019):
        #convert data into csv file with year in new column
        artistSR = json_normalize(data, str(year), errors='ignore', record_prefix = ' ')
        artistSR['year'] = year
        with open("artistSongRank.csv", 'a') as outFile:
            artistSR.to_csv(outFile, header = ['artist(s)', 'rank', 'song', 'year'])
        artistSongRank = artistSongRank.append(artistSR)
    
    artistSongRank['gender'] = "" #new column in dataframe to hold gender value
    
    for col in artistGenderData.columns:
        artistSongRank.loc[artistSongRank[' artist(s)'] == col, 'gender'] = artistGenderData[col].iloc[0]  
   
    artistSongRank.to_csv("artistGenderRankData.csv")
        
    return artistSongRank
                
#readData
            
def descriptiveAnalytics(myData):
    artist = myData[' artist(s)'].mode()
    gender = myData['gender'].mode()
    print("The artist that is in the top 100 the most over the past 56 years is: " + artist[0])
    print("The gender that is in the top 100 the most over the past 56 years is: " + gender[0])
    print("-----------------------------------------------------------------------------------")
    
    for year in range(1963, 2019):
        artistlist = []
        genderlist = []
        for index, row in myData.iterrows():
            if row['year'] == year:
                artistlist.append(row[' artist(s)'])
                genderlist.append(row['gender'])
        print("The artist that is in the top 100 the most in year " + str(year) + " is: " + most_frequent(artistlist))
        print("The gender that is most frequent in the top 100 in year " + str(year) + " is: " + most_frequent(genderlist))
#descriptiveAnalytics

def most_frequent(listlist):
    count = 0
    most = listlist[0]
    
    for i in listlist:
        freq = listlist.count(i)
        if (freq > count):
            count = freq
            most = i
            
    return most
#most_frequent

if __name__ == '__main__':
    main()
