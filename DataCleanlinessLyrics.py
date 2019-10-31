#Project 1: for cleaning the data for lyrical data
import pandas as pd
import numpy as np

def main():
    myData = pd.read_csv('LyricData.csv', encoding='latin1') #reading csv file directly into panda object
    
    missingValues(myData) #call to find the missing values of the pandas dataframe
    
def missingValues(myData):
    for i in list(myData.columns.values): #iterate through each column
        print("The fraction of missing values for " + str(i) + " is: " + str(columnValues(myData, i)/len(myData)) + "\n") #print the fraction of missing values for each column
    print("The fraction of noise values are: " + str(noiseValues(myData)/(len(myData)*3))) #print the fraction of noise values in the dataframe
#missingValues
    
def columnValues(myData, column): #pass in the dataFrame and the column name
    count = 0
    for i in myData[column]: #for each item in the column
        if i == "": #if the item is empty
            count = count + 1 #add 1 to count
    return count #return the count

def noiseValues(myData):
    noisecount = 0
    for index, row in myData.iterrows(): #iterate through each row
        for item in row.iteritems(): #iterate through each item in the row
            if type(item) != tuple: #if the item is not equal to a tuple
                noisecount = noisecount + 1 #increment the noisecount by 1
    return noisecount #return the noisecount
#columnValues        

main()
