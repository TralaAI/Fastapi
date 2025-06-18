# Endpoints
Two endpoints have been made:
- /predict --> Will return predicted litter amounts of 5 types for a certain amount of days into the future. 
- /retrain --> Will put new data into the csv the model has acces to and then make it retrain outputting a new .pkl file that used the new data for its training

Expected body for /predict endpoint:
- modelIndex : "0-4" --> 0 = developing 1, = Sensoring group, 2 = generated_city, 3 = generated_industrial, 4 = generated_suburbs (What model you want to get the prediction from?)
- inputs : [{input array}]
- - day_of_week : 0-6 (monday to sunday)
- - month : 1-12 (januari to december)
- - holiday : bool (true or false holiday)
- - weather : 1-6 ('snowy': 1, 'stormy': 2, 'rainy': 3, 'misty': 4, 'cloudy': 5, 'sunny':6)
- - temperature_celcius : -100 - 100 (temperature rounded to whole number)
- - is_weekend : (is it sat or sun? - Could be derived from day_of_week but column of its own makes it easier for AI-model)
- - label: string (determine on your own to make it easier to seperate returns)

Example:
```
{
  "modelIndex": "0",
  "inputs": [
    {
      "day_of_week": 1,
      "month": 6,
      "holiday": false,
      "weather": 2,
      "temperature_celcius": 20,
      "is_weekend": false,
      "label": "day1"
    },
    {
      "day_of_week": 2,
      "month": 6,
      "holiday": false,
      "weather": 3,
      "temperature_celcius": 19,
      "is_weekend": false,
      "label": "day2"
    },
    {
      "day_of_week": 5,
      "month": 6,
      "holiday": true,
      "weather": 4,
      "temperature_celcius": 22,
      "is_weekend": true,
      "label": "day3"
    }
  ]
}
```

Return from /predict endpoint:

array of objects containing the ealier sent label as wel as 5 categories of litter and their predicted amounts:
```
{
    "predictions": [
        {
            "day1": {
                "plastic": 0.7750469376939965,
                "paper": 0.7847388144446968,
                "metal": 1.2259807706866532,
                "glass": 2.0541062911651147,
                "organic": 2.2382871526989176
            }
        },
        {
            "day2": {
                "plastic": 1.0564339446470412,
                "paper": 0.7574026507877785,
                "metal": 1.8400461568275108,
                "glass": 1.2374766426486739,
                "organic": 2.380200130366612
            }
        },
        {
            "day3": {
                "plastic": 1.894920634920635,
                "paper": 1.4498412698412697,
                "metal": 2.284761904761905,
                "glass": 3.363809523809524,
                "organic": 5.203492063492064
            }
        }
    ]
}
```

Expected input for /retrain endpoint
- cameraLocation : 0-4 --> What location is the new data coming from? (0 = developing 1, = Sensoring group, 2 = generated_city, 3 = generated_industrial, 4 = generated_suburbs)
- data : [{array of new data}]
- - timestamp : !YYYY-MM-DD HH:MM:SS! --> When was this data recorded? !!Make sure to use this time format!!
- - detected_object : "litterType" --> 1 of 5 litter types: plastic, paper, metal, glass & organic
- - holiday : bool
- - weather : "weatherType" --> 1 of 6 weathertypes: snowy, stormy, rainy, misty, cloudy & sunny
- - temperature_celcius : -100 - 100 (temperature at the time of record)

Example:
```
{
  "cameraLocation": "0",
  "data": [
    {
      "timestamp": "2025-06-17 10:00:00",
      "detected_object": "plastic",
      "holiday": false,
      "weather": "sunny",
      "temperature_celsius": 21
    },
    {
      "timestamp": "2025-06-17 12:30:00",
      "detected_object": "glass",
      "holiday": false,
      "weather": "cloudy",
      "temperature_celsius": 20
    },
    {
      "timestamp": "2025-06-17 15:00:00",
      "detected_object": "paper",
      "holiday": true,
      "weather": "rain",
      "temperature_celsius": 18
    }
  ]
}
```

return from /retrain endpoint:
- status : ok/error
- added_rows : int (how many new recors were added?)
- exec_output : {} --> The RMSE of the new generated model

Example:
```
{
    "status": "success",
    "added_rows": 3,
    "exec_output": {
        "stdout": "Train RMSE: 3.9067344264730184\nTest RMSE: 4.049872082308912\n",
        "stderr": "",
        "message": "Model.py executed successfully."
    }
}
```