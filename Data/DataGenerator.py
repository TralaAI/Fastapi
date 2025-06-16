import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Parameters
start_date = '2025-01-01'
end_date = '2025-12-31'
weather_types = ['rainy', 'cloudy', 'sunny', 'stormy', 'misty']
litter_types = ['paper', 'plastic', 'organic', 'metal', 'glass']

# Full Dutch holidays 2025 including regional & cultural
holidays = [
    '2025-01-01',  # New Year's Day
    '2025-02-24',  # Carnaval Monday
    '2025-02-25',  # Carnaval Tuesday
    '2025-02-26',  # Ash Wednesday
    '2025-04-18',  # Good Friday
    '2025-04-20',  # Easter Sunday
    '2025-04-21',  # Easter Monday
    '2025-04-27',  # King's Day
    '2025-05-05',  # Liberation Day
    '2025-05-29',  # Ascension Day
    '2025-06-08',  # Pentecost Sunday
    '2025-06-09',  # Pentecost Monday
    '2025-11-11',  # Sint-Maarten
    '2025-12-05',  # Sinterklaasavond
    '2025-12-25',  # Christmas Day
    '2025-12-26',  # Boxing Day
    '2025-12-31'   # New Year's Eve
]

holidays = set(pd.to_datetime(holidays))

def is_weekend(date):
    return date.weekday() >= 5

def assign_weather(date):
    month = date.month
    if month in [12, 1, 2]:
        probs = [0.3, 0.3, 0.1, 0.2, 0.1]
    elif month in [6, 7, 8]:
        probs = [0.1, 0.4, 0.4, 0.05, 0.05]
    else:
        probs = [0.2, 0.4, 0.3, 0.05, 0.05]
    return np.random.choice(weather_types, p=probs)

def litter_rate(date, weather):
    base_rate = 13
    weekend_bonus = 0.3 if is_weekend(date) else 0
    holiday_bonus = 0.7 if date in holidays else 0
    weather_effect = {
        'stormy': -0.6,
        'rainy': -0.4,
        'cloudy': 0,
        'misty': -0.1,
        'sunny': 0.3
    }
    rate = base_rate * (1 + weekend_bonus + holiday_bonus + weather_effect.get(weather, 0))
    return max(1, int(rate))

def pick_litter_type(weather):
    if weather == 'stormy':
        probs = [0.1, 0.1, 0.5, 0.1, 0.2]
    elif weather == 'sunny':
        probs = [0.25, 0.3, 0.2, 0.15, 0.1]
    else:
        probs = [0.2, 0.25, 0.25, 0.15, 0.15]
    return np.random.choice(litter_types, p=probs)

all_rows = []
current_id = 1
date_range = pd.date_range(start_date, end_date, freq='D')

for date in date_range:
    weather = assign_weather(date)
    holiday_flag = 1 if date in holidays else 0
    n_pieces = litter_rate(date, weather)

    for _ in range(n_pieces):
        random_seconds = random.randint(0, 86399)
        timestamp = datetime.combine(date, datetime.min.time()) + timedelta(seconds=random_seconds)

        litter = pick_litter_type(weather)

        all_rows.append({
            'id': current_id,
            'detected_object': litter,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'weather': weather,
            'holiday': holiday_flag
        })
        current_id += 1

df = pd.DataFrame(all_rows)
df.to_csv('generated_litter_data_2025_full_year_all_holidays.csv', index=False)
print("Dataset generated and saved as 'generated_litter_data_2025_full_year_all_holidays.csv'")
