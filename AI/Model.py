import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import numpy as np

afval = pd.read_csv('../Data/afval_voorbeeld_3900.csv')

# Instead of detected_object being a class. I made their classes columns with integer values. Next step is to group them per time interval. 
afval_encoded = pd.get_dummies(afval, columns=['detected_object'], dtype=int)

# Convert timestamp to datetime and extract date
afval_encoded['timestamp'] = pd.to_datetime(afval_encoded['timestamp'])
afval_encoded['date'] = afval_encoded['timestamp'].dt.date

# Mapping the weather data. Rainy 1, Cloudy 2, Sunny 3, Stormy 4, Misty 5
weather_mapping = {'rainy': 1, 'cloudy': 2, 'sunny': 3, 'stormy': 4, 'misty': 5}
afval_encoded['weather'] = afval_encoded['weather'].map(weather_mapping)

# Grouping my data per day
daily_counts = afval_encoded.groupby('date').agg({
    'holiday': lambda x: 1 if (x==1).any() else 0,  # total number of detections that are on holiday = fine
    'weather': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,  # take most frequent weather (mode)
    'detected_object_glass': 'sum',
    'detected_object_metal': 'sum',
    'detected_object_organic': 'sum',
    'detected_object_paper': 'sum',
    'detected_object_plastic': 'sum'
}).reset_index()

#Adding columns with summed litter counts
daily_counts['litter_total'] = daily_counts[
    ['detected_object_glass', 'detected_object_metal', 'detected_object_organic', 'detected_object_paper', 'detected_object_plastic']
].sum(axis=1)

# Convert 'date' back to datetime to extract day_of_week and month features
daily_counts['date'] = pd.to_datetime(daily_counts['date'])
daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek  
daily_counts['month'] = daily_counts['date'].dt.month

print(daily_counts)
features = ['detected_object_glass', 'detected_object_metal', 'detected_object_organic', 'detected_object_paper', 'detected_object_plastic']


x = daily_counts[['day_of_week', 'month', 'holiday', 'weather']] # Our training features
y = daily_counts[features]  # Our target variable

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

dt = DecisionTreeRegressor(max_depth=4)
dt.fit(x_train, y_train)

#---------SCHOOL FUNCTIONS-----------
def calculate_rmse(predictions, actuals):
    if(len(predictions) != len(actuals)):
        raise Exception("The amount of predictions did not equal the amount of actuals")
    
    return (((predictions - actuals) ** 2).sum() / len(actuals)) ** (1/2)
#---------SCHOOL FUNCTIONS-----------

predict_train = dt.predict(x_train)
predict_test = dt.predict(x_test)

# print(predict_train)
# print(predict_test)

rmse_train = calculate_rmse(predict_train, y_train.values)
rmse_test = calculate_rmse(predict_test, y_test.values)

print(f"Train RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")