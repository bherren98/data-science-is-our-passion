#Project 3 1 visualization and 1 interaction

import pandas as pd
import plotly.graph_objects as go

def main():
    myData = readData()
    visualization(myData)

def readData():
    myData = pd.read_csv('artistGenderRankData.csv')
    return myData
#readData

def visualization(myData):
    genderList = []
    for year in range(1963, 2019):
        for index, row in myData.iterrows():
            if row['year'] == year:
                genderList.append(row['gender'])
    femaleCount = genderList.count('female')
    maleCount = genderList.count('male')
    unknownCount = genderList.count('unknown')
        
    labels = ['Female', 'Male', 'Unknown']
    values = [femaleCount, maleCount, unknownCount]
        
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.show()
#visualization
    
if __name__ == '__main__':
    main()

