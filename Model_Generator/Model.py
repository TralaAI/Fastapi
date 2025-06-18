import graphviz
import sys
import numpy as np
import pandas as pd
import joblib as jb
from pathlib import Path
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

parameter = '0'

if len(sys.argv) > 1:
    parameter = sys.argv[1]

match parameter:
    case '0':
        OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'developing_phase_tree.pkl'
        DATA_PATH = BASE_DIR.parent / 'Data' / 'developing_data.csv'
    case '1':
        OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'sensoring_group_tree.pkl'
        DATA_PATH = BASE_DIR.parent / 'Data' / 'sensoring_data.csv'
    case '2':
        OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'generated_city_tree.pkl'
        DATA_PATH = BASE_DIR.parent / 'Data' / 'city_data.csv'
    case '3':
        OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'generated_industrial_tree.pkl'
        DATA_PATH = BASE_DIR.parent / 'Data' / 'industrial_data.csv'
    case '4':
        OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'generated_suburbs_tree.pkl'
        DATA_PATH = BASE_DIR.parent / 'Data' / 'suburbs_data.csv'
    case _:
        print(f"Invalid parameter: {parameter}")
        sys.exit(1)


afval = pd.read_csv(DATA_PATH)

# Instead of detected_object being a class. I made their classes columns with integer values. Next step is to group them per time interval. 
afval_encoded = pd.get_dummies(afval, columns=['detected_object'], dtype=int)

# Convert timestamp to datetime and extract date
afval_encoded['timestamp'] = pd.to_datetime(afval_encoded['timestamp'])
afval_encoded['date'] = afval_encoded['timestamp'].dt.date

# Mapping the weather data. Rainy 1, Cloudy 2, Sunny 3, Stormy 4, Misty 5
weather_mapping = {'snowy': 1, 'stormy': 2, 'rainy': 3, 'misty': 4, 'cloudy': 5, 'sunny':6}
afval_encoded['weather'] = afval_encoded['weather'].map(weather_mapping)

# Grouping my data per day
daily_counts = afval_encoded.groupby('date').agg({
    'holiday': lambda x: 1 if (x==1).any() else 0,  #
    'weather': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan, 
    'detected_object_glass': 'sum',
    'temperature_celsius': 'mean',
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
daily_counts['is_weekend'] = daily_counts['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


print(daily_counts)
features = ['detected_object_glass', 'detected_object_metal', 'detected_object_organic', 'detected_object_paper', 'detected_object_plastic']


x = daily_counts[['day_of_week', 'month', 'holiday', 'weather', 'temperature_celsius', 'is_weekend']] # Our training features
y = daily_counts[features]  #target variable

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# dt = DecisionTreeRegressor(max_depth=4)
dt = RandomForestRegressor(n_estimators=5, max_depth=3) 
dt.fit(x_train, y_train)

#---------SCHOOL FUNCTIONS-----------
def calculate_rmse(predictions, actuals):
    if(len(predictions) != len(actuals)):
        raise Exception("The amount of predictions did not equal the amount of actuals")
    
    return (((predictions - actuals) ** 2).sum() / len(actuals)) ** (1/2)

def plot_tree_regression(model, features):
    # Generate plot data
    output_path = BASE_DIR / "decision_tree"
    dot_data = tree.export_graphviz(model, out_file=None, 
                          feature_names=features,  
                          filled=True, rounded=True,  
                          special_characters=True)  

    # Turn into graph using graphviz
    graph = graphviz.Source(dot_data)  

    # Write out a pdf
    graph.render(output_path)

    # Display in the notebook
    return graph 

#---------SCHOOL FUNCTIONS-----------
# plot_tree_regression(dt, ['day_of_week', 'month', 'holiday', 'weather', 'temperature_celsius', 'is_weekend'])


predict_train = dt.predict(x_train)
predict_test = dt.predict(x_test)

# print(predict_train)
# print(predict_test)

rmse_train = calculate_rmse(predict_train, y_train.values)
rmse_test = calculate_rmse(predict_test, y_test.values)

print(f"Train RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")

jb.dump(dt, OUTPUT_PATH)