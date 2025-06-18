import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / '2_city_data.csv'

# Parameters
start_date = '2025-01-01'
end_date = '2025-12-31'
weather_types = ['rainy', 'cloudy', 'sunny', 'stormy', 'misty', 'snowy']
litter_types = ['paper', 'plastic', 'organic', 'metal', 'glass']

# Custom litter distribution
custom_litter_distribution = {
    'paper': 0.20,
    'plastic': 0.35,
    'organic': 0.20,
    'metal': 0.125,
    'glass': 0.125
}

# Dutch holidays 2025
holidays = {
    pd.Timestamp(d) for d in [
        '2025-01-01', '2025-02-24', '2025-02-25', '2025-02-26', '2025-04-18',
        '2025-04-20', '2025-04-21', '2025-04-27', '2025-05-05', '2025-05-29',
        '2025-06-08', '2025-06-09', '2025-11-11', '2025-12-05', '2025-12-25',
        '2025-12-26', '2025-12-31'
    ]
}

# Monthly average temperatures (°C)
monthly_avg_temps = {
    1: 3, 2: 4, 3: 7, 4: 10, 5: 15, 6: 18,
    7: 21, 8: 23, 9: 19, 10: 14, 11: 8, 12: 4
}

# Temperature bounds for realism (min, max) for each month
month_temp_bounds = {
    1: (-5, 8),
    2: (-3, 9),
    3: (0, 12),
    4: (3, 16),
    5: (8, 21),
    6: (12, 25),
    7: (15, 28),
    8: (15, 28),
    9: (10, 23),
    10: (5, 18),
    11: (0, 13),
    12: (-3, 9)  
}

# Easy-to-adjust effect modifiers for weather and temperature
WEATHER_EFFECTS = {
    'stormy': -0.9,
    'rainy': -0.5,
    'cloudy': 0.4,
    'misty': -0.2,
    'sunny': 0.6,   # boost sunny to be 100% more litter
    'snowy': -0.9
}

TEMPERATURE_EFFECTS = {
    'high': {
        'threshold': 20,
        'modifier': 0.3 
    },
    'low': {
        'threshold': 5,
        'modifier': -0.3
    },
    'moderate': 0
}


def is_weekend(date):
    return date.weekday() >= 5


def assign_weather(date):
    month = date.month
    if month in [12, 1, 2]:  # Winter months with snow possible
        probs = [0.25, 0.25, 0.05, 0.15, 0.1, 0.2]  # snowy: 20% chance in winter
    elif month in [6, 7, 8]:  # Summer
        probs = [0.1, 0.4, 0.4, 0.05, 0.05, 0.0]  # no snowy weather in summer
    else:  # Spring & Autumn
        probs = [0.2, 0.4, 0.3, 0.05, 0.05, 0.0]  # no snowy weather
    return np.random.choice(weather_types, p=probs)


def assign_temperature(date):
    avg_temp = monthly_avg_temps[date.month]
    min_temp, max_temp = month_temp_bounds[date.month]
    std_dev = 2 if date.month in [12, 1, 2] else 3
    temp = np.random.normal(loc=avg_temp, scale=std_dev)
    temp = np.clip(temp, min_temp, max_temp)
    return round(temp)


def temperature_modifier(temperature):
    if temperature >= TEMPERATURE_EFFECTS['high']['threshold']:
        return TEMPERATURE_EFFECTS['high']['modifier']
    elif temperature <= TEMPERATURE_EFFECTS['low']['threshold']:
        return TEMPERATURE_EFFECTS['low']['modifier']
    else:
        return TEMPERATURE_EFFECTS['moderate']


def litter_rate(date, weather, temperature):
    base_rate = 15
    weekend_bonus = 0.3 if is_weekend(date) else 0
    holiday_bonus = 0.7 if date in holidays else 0
    weather_bonus = WEATHER_EFFECTS.get(weather, 0)
    temp_bonus = temperature_modifier(temperature)
    rate = base_rate * (1 + weekend_bonus + holiday_bonus + weather_bonus + temp_bonus)
    return max(1, int(rate))


def pick_litter_type():
    litter_probs = [custom_litter_distribution[lt] for lt in litter_types]
    return np.random.choice(litter_types, p=litter_probs)


# Generate dataset
all_rows = []
current_id = 1
date_range = pd.date_range(start_date, end_date, freq='D')

for date in date_range:
    weather = assign_weather(date)
    temperature = assign_temperature(date)
    holiday_flag = 1 if date in holidays else 0
    n_pieces = litter_rate(date, weather, temperature)

    for _ in range(n_pieces):
        random_seconds = random.randint(0, 86399)
        timestamp = datetime.combine(date, datetime.min.time()) + timedelta(seconds=random_seconds)
        litter = pick_litter_type()

        all_rows.append({
            'id': current_id,
            'detected_object': litter,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'weather': weather,
            'temperature_celsius': temperature,
            'holiday': holiday_flag
        })
        current_id += 1

df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_PATH, index=False)
print("Dataset generated and saved to", OUTPUT_PATH)
