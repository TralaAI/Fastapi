from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn import tree
from pathlib import Path
from fpdf import FPDF
import joblib as jb
import pandas as pd
import numpy as np
import graphviz
import os

def train_and_save_model(parameter='0'):
    BASE_DIR = Path(__file__).resolve().parent

    match parameter:
        case '0':
            OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'developing_phase_tree.pkl'
            DRAW_PATH = BASE_DIR / 'Plot_Developing' / 'developing_phase_tree'
            LOCATION = 'development'
        case '1':
            OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'sensoring_group_tree.pkl'
            DRAW_PATH = BASE_DIR / 'Plot_Sensoring' / 'sensoring_group_tree'
            LOCATION = 'sensoring'
        case '2':
            OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'generated_city_tree.pkl'
            DRAW_PATH = BASE_DIR / 'Plot_City' / 'generated_city_tree'
            LOCATION = 'city'
        case '3':
            OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'generated_industrial_tree.pkl'
            DRAW_PATH = BASE_DIR / 'Plot_Industrial' / 'generated_industrial_tree'
            LOCATION = 'industrial'
        case '4':
            OUTPUT_PATH = BASE_DIR.parent / 'AI_Models' / 'generated_suburbs_tree.pkl'
            DRAW_PATH = BASE_DIR / 'Plot_Suburbs' / 'generated_suburbs_tree'
            LOCATION = 'suburbs'
        case _:
            print(f"Invalid parameter: {parameter}")
            return

    load_dotenv()
    connection_string = os.getenv("connStr")
    engine = create_engine(connection_string)
    query = f"SELECT * FROM litters WHERE location='{LOCATION}'"
    afval = pd.read_sql(query, engine)
    afval_encoded = pd.get_dummies(afval, columns=['Type'], dtype=int)
    afval_encoded['timestamp'] = pd.to_datetime(afval_encoded['TimeStamp'])
    afval_encoded['date'] = afval_encoded['timestamp'].dt.date
    weather_mapping = {'snowy': 1, 'stormy': 2, 'rainy': 3, 'misty': 4, 'cloudy': 5, 'sunny': 6}
    afval_encoded['weather'] = afval_encoded['Weather'].map(weather_mapping)
    daily_counts = afval_encoded.groupby('date').agg({
        'IsHoliday': lambda x: 1 if (x == 1).any() else 0,
        'weather': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'Type_glass': 'sum',
        'Temperature': 'mean',
        'Type_metal': 'sum',
        'Type_organic': 'sum',
        'Type_paper': 'sum',
        'Type_plastic': 'sum'
    }).reset_index()
    daily_counts['litter_total'] = daily_counts[
        ['Type_glass', 'Type_metal', 'Type_organic', 'Type_paper', 'Type_plastic']
    ].sum(axis=1)
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
    daily_counts['month'] = daily_counts['date'].dt.month
    daily_counts['is_weekend'] = daily_counts['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    features = ['Type_glass', 'Type_metal', 'Type_organic', 'Type_paper', 'Type_plastic']
    x = daily_counts[['day_of_week', 'month', 'IsHoliday', 'weather', 'Temperature', 'is_weekend']]
    y = daily_counts[features]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    dt = RandomForestRegressor(n_estimators=5, max_depth=3)
    dt.fit(x_train, y_train)

    def calculate_rmse(predictions, actuals):
        if len(predictions) != len(actuals):
            raise Exception("The amount of predictions did not equal the amount of actuals")
        return (((predictions - actuals) ** 2).sum() / len(actuals)) ** 0.5

    def plot_tree_regression(model, features, output_file=DRAW_PATH):
        if not hasattr(model, 'estimators_'):
            raise ValueError("The model does not have estimators_ attribute. Is it a RandomForestRegressor?")
        pdf = FPDF()
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path
        base_path = output_path.with_suffix('')
        for i, tree_model in enumerate(model.estimators_):
            dot_data = tree.export_graphviz(
                tree_model,
                out_file=None,
                feature_names=features,
                filled=True,
                rounded=True,
                special_characters=True
            )
            graph = graphviz.Source(dot_data)
            image_file = base_path.with_name(f"{base_path.name}_{i+1}.png")
            graph.render(filename=str(image_file.with_suffix('')), format='png')
            pdf.add_page()
            pdf.image(str(image_file), x=10, y=10, w=180)
        pdf.output(str(output_path.with_suffix('.pdf')))
        return graph

    predict_train = dt.predict(x_train)
    predict_test = dt.predict(x_test)
    rmse_train = calculate_rmse(predict_train, y_train.values)
    rmse_test = calculate_rmse(predict_test, y_test.values)
    print(f"Train RMSE: {rmse_train}")
    print(f"Test RMSE: {rmse_test}")
    jb.dump(dt, OUTPUT_PATH)
    # Optionally plot trees:
    # plot_tree_regression(dt, x.columns.tolist(), DRAW_PATH)

# Example usage:
# train_and_save_model('0')
