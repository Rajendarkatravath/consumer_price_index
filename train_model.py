import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the data
data = pd.read_csv('E:/data_science_projects/Nigeria Economy/dataset/1960_onwards.csv')

# Define the features and target
independent_feature = 'Consumer price index (2010 = 100)'
dependent_features = [
    'GDP (current LCU)', 'Official exchange rate (LCU per US$, period average)', 'Population, total',
    'Cumulative crude oil production up to and including year', 'Narrow Money', 'Credit to Private Sector',
    'Demand Deposits', 'Population ages 65 and above (% of total population)', 'Money Supply M2',
    'Population, female', 'Quasi Money', 'Bank Reserves', 'Livestock production index (2014-2016 = 100)',
    'Net Foreign Assets', 'GDP (constant LCU)'
]

X = data[dependent_features]
y = data[independent_feature]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the CatBoostRegressor
model = CatBoostRegressor(verbose=0)

# Train the model
model.fit(X_train, y_train)

# Save the model to disk
pickle.dump(model, open('./model.sav', 'wb'))
print("Model saved as 'model.sav'")
