from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any
from sklearn import tree
from fpdf import FPDF
import pandas as pd
import numpy as np
import joblib as jb
import graphviz
import os

BASE_DIR = Path(__file__).resolve().parent

# --- Utility Functions ---

def calculate_rmse(predictions, actuals):
    """Calculate Root Mean Squared Error."""
    if len(predictions) != len(actuals):
        raise Exception("The amount of predictions did not equal the amount of actuals")
    return (((predictions - actuals) ** 2).sum() / len(actuals)) ** 0.5

def plot_tree_regression(model, features, output_file):
    """Plot each tree in the RandomForestRegressor and save as a PDF."""
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

# --- Main Model Training Function ---

def train_and_save_model(parameter: int = 0) -> Dict[str, Any]:
    """
    Train a RandomForestRegressor on litter data for a specific camera,
    save the model, and optionally plot the trees.
    """
    # --- Directory Setup ---
    ai_models_dir = BASE_DIR / 'AI_Models'
    ai_models_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = BASE_DIR / 'Model_Plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = ai_models_dir / f"Camera{parameter}_tree.pkl"

    # --- Plot Path Selection ---
    plot_paths = {
        1: plot_dir / 'Plot_Developing' / 'developing_phase_tree',
        2: plot_dir / 'Plot_Sensoring' / 'sensoring_group_tree',
        3: plot_dir / 'Plot_City' / 'generated_city_tree',
        4: plot_dir / 'Plot_Industrial' / 'generated_industrial_tree',
        5: plot_dir / 'Plot_Suburbs' / 'generated_suburbs_tree'
    }
    DRAW_PATH = plot_paths.get(parameter, plot_dir / 'Plot_Others' / 'others_tree')

    # --- Database Connection ---
    load_dotenv()
    connection_string = os.getenv("connStr")
    if not connection_string:
        raise ValueError("Database connection string (connStr) is not set in environment variables.")
    engine = create_engine(connection_string)

    # --- Data Loading ---
    CAMERA_ID = parameter
    query = f"SELECT * FROM litters WHERE CameraId='{CAMERA_ID}'"
    afval: pd.DataFrame = pd.read_sql(query, engine)

    # --- Data Preprocessing ---
    if 'Type' not in afval.columns:
        raise KeyError("Column 'Type' does not exist in the database table 'litters'.")
    afval_encoded: pd.DataFrame = pd.get_dummies(afval, columns=['Type'], dtype=int)
    expected_types = ['Type_glass', 'Type_metal', 'Type_organic', 'Type_paper', 'Type_plastic']
    for col in expected_types:
        if col not in afval_encoded.columns:
            afval_encoded[col] = 0

    afval_encoded['timestamp'] = pd.to_datetime(afval_encoded['TimeStamp'])
    afval_encoded['date'] = afval_encoded['timestamp'].dt.date

    weather_mapping = {'snowy': 1, 'stormy': 2, 'rainy': 3, 'misty': 4, 'cloudy': 5, 'sunny': 6}
    afval_encoded['weather'] = afval_encoded['Weather'].map(weather_mapping)

    # --- Aggregation ---
    daily_counts: pd.DataFrame = afval_encoded.groupby('date').agg({
        'IsHoliday': lambda x: 1 if (x == 1).any() else 0,
        'weather': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'Type_glass': 'sum',
        'Temperature': 'mean',
        'Type_metal': 'sum',
        'Type_organic': 'sum',
        'Type_paper': 'sum',
        'Type_plastic': 'sum'
    }).reset_index()

    daily_counts['litter_total'] = daily_counts[expected_types].sum(axis=1)
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
    daily_counts['month'] = daily_counts['date'].dt.month
    daily_counts['is_weekend'] = daily_counts['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # --- Feature Selection ---
    features = expected_types
    x = daily_counts[['day_of_week', 'month', 'IsHoliday', 'weather', 'Temperature', 'is_weekend']]
    y = daily_counts[features]

    # --- Model Training ---
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=5, max_depth=3)
    model.fit(x_train, y_train)

    # --- Evaluation ---
    predict_train = model.predict(x_train)
    predict_test = model.predict(x_test)
    rmse_train = calculate_rmse(predict_train, y_train.values)
    rmse_test = calculate_rmse(predict_test, y_test.values)

    print(f"Model {CAMERA_ID}")
    print(f"Train RMSE: {rmse_train}")
    print(f"Test RMSE: {rmse_test}\n")

    # --- Save Model ---
    jb.dump(model, OUTPUT_PATH)

    # --- Plot Trees (if not in Production) ---
    environment = os.getenv("Environment")
    if environment != "Production":
        try:
            plot_tree_regression(model, x.columns.tolist(), DRAW_PATH)
        except Exception as e:
            print(f"Failed to plot tree for Camera {CAMERA_ID}: {e}")

    return {
        "camera": parameter,
        "rmse": (rmse_train + rmse_test) / 2
    }