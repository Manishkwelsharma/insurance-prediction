## Insurance Premium Predictor 💼
This application predicts insurance premiums based on the user's age using a linear regression model. The application was developed with Python and uses scikit-learn for the model, pandas for data handling, and Streamlit for the user interface.
## Live Demo:
Link - http://localhost:8502
## Features:
Takes the user's age as input (between 1 and 120 years).
Predicts the insurance premium using a linear regression model trained on historical data.
Scales input features using MinMaxScaler for better performance.
## Dataset:
The dataset used for training the model is expected to have the following structure:

Age: The age of the user (feature used for prediction).
Premium: The corresponding insurance premium (target).
The dataset must be in CSV format and should have at least these two columns.

## Model:
The model used in this application is a simple linear regression model. It was trained on 80% of the data and tested on the remaining 20%.

## Prerequisites:
To run this project locally, ensure you have the following installed:

Python 3.8 or later
pandas
scikit-learn
streamlit
numpy