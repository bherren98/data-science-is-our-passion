from pandas.io.json import json_normalize
import pandas as pd
import json
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def main():
    myData = readFile()
    ttest(myData)
    linearRegression(myData)
    
def readFile():
    myData = pd.read_csv("genre_correlation_11-08-2019_121532.csv")
    
    return myData
#readFile
    
def ttest(myData):
    
    print(ttest_ind(myData['r&b_exact'], myData['rock_exact']))
    
#ttest
    
def linearRegression(myData):
    X = myData['r&b_exact'].values.reshape(-1,1)
    y = myData['rock_exact'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    print("Regressor intercept: " + str(regressor.intercept_))
    print("Regressor Coefficient: " + str(regressor.coef_))
    
    y_pred = regressor.predict(X_test)
    
    newData = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    
    plt.scatter(X_test, y_test,  color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show()
#linearRegression
    
if __name__ == '__main__':
    main()