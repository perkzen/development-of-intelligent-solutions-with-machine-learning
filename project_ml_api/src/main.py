from datetime import datetime

import numpy as np
from fastapi import FastAPI

from src.models.air_quality import AirQuality
from src.utils.create_sequence import create_sequences
from src.utils.load_object import load_model_object, load_scaler_object

app = FastAPI()

model = load_model_object("model_mse_35.81.h5")
scaler = load_scaler_object("scaler.pkl")


@app.get("/")
async def root():
    date = datetime.now()
    return {"status": "ok", "date": date}


@app.post("/predict")
async def predict(data: list[AirQuality]):
    data_values = [[item.pm10, item.pm25, item.pm25_o3, item.pm25_no2] for item in data]
    scaled_data = scaler.transform(data_values)

    feature_cols = list(range(len(data_values[0])))
    window_size = 48

    X = create_sequences(scaled_data, window_size, feature_cols)

    prediction = model.predict(X)

    prediction_copies_array = np.repeat(prediction, len(feature_cols), axis=-1)
    prediction_reshaped = np.reshape(prediction_copies_array, (len(prediction), len(feature_cols)))
    prediction = scaler.inverse_transform(prediction_reshaped)[:, 0]

    return {"prediction": prediction.tolist()[0]}
