import streamlit as st
import pandas as pd
import pickle

# Load the model from disk
model = pickle.load(open('./model.sav', 'rb'))

# Define the feature names
dependent_features = [
    'GDP (current LCU)', 'Official exchange rate (LCU per US$, period average)', 'Population, total',
    'Cumulative crude oil production up to and including year', 'Narrow Money', 'Credit to Private Sector',
    'Demand Deposits', 'Population ages 65 and above (% of total population)', 'Money Supply M2',
    'Population, female', 'Quasi Money', 'Bank Reserves', 'Livestock production index (2014-2016 = 100)',
    'Net Foreign Assets', 'GDP (constant LCU)'
]

# Streamlit app
st.title('Consumer Price Index Prediction')

# Input fields for user to enter feature values
inputs = {}
for feature in dependent_features:
    inputs[feature] = st.number_input(f'Enter {feature}', value=0.0)

# Predict button
if st.button('Predict'):
    # Create a DataFrame for the inputs
    input_data = pd.DataFrame([inputs])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f'The predicted Consumer Price Index (2010 = 100) is: {prediction[0]}')

st.write('Model Feature Importances:')
importances = model.get_feature_importance()
feature_importances = pd.DataFrame({'Feature': dependent_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
st.bar_chart(feature_importances.set_index('Feature'))
