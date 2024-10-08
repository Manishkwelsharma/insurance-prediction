# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data():
    data = pd.read_csv('data/insurance_data.csv')
    return data

def train_model(data):
    X = data[['Age']]
    y = data['Premium']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model
