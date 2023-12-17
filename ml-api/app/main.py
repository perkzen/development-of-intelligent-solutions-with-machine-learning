from typing import List
from fastapi import FastAPI
from app.models.image import ImageData
from app.models.weather import WeatherData, ExtendedWeatherData
from app.utils.load_models import load_models_task4, load_models_task6, load_models_task5
from app.utils.sliding_window import sliding_window
from datetime import datetime
import numpy as np
import base64
from PIL import Image
import io
from tensorflow.keras.preprocessing.image import img_to_array

model_task4, scaler_task4 = load_models_task4()
model_task5 = load_models_task5()
model_task6, scaler_task6 = load_models_task6()

app = FastAPI()


@app.get("/")
async def root():
    return \
        {
            "message": "ML API",
            "description": "This is the API for the ML model for the task 4 and 6 of the course",
            "timestamp": datetime.now(),
            "status": "OK"
        }


@app.post("/predict/task4")
async def predict(data: WeatherData):
    input_features = [
        data.apparent_temperature_difference,
        data.apparent_temperature,
        data.temperature,
        data.hour,
        data.precipitation_probability
    ]

    features = np.array(input_features).reshape(1, -1)
    features = scaler_task4.transform(features)

    prediction = model_task4.predict(features)
    return {"prediction": prediction.item()}


@app.post("/predict/task5")
def predict_image(data: ImageData):
    img_data = base64.b64decode(data.image_base64)
    image = Image.open(io.BytesIO(img_data)).convert('L')
    image = image.resize((100, 100))

    image_array = img_to_array(image)
    image_array = image_array / 255.0

    prediction = model_task5.predict(image_array.reshape(1, 100, 100, 1))
    prediction = np.argmax(prediction, axis=1)
    classes = ['Paper', 'Rock', 'Scissors']
    pred_class = [classes[i] for i in prediction]

    return {"prediction": pred_class[0]}


@app.post("/predict/task6")
async def predict_line(data: List[ExtendedWeatherData]):
    data = [d.dict() for d in data]
    data = sorted(data, key=lambda x: x['date'])
    data = [d['available_bike_stands'] for d in data]
    data = np.array(data).reshape(-1, 1)
    data = scaler_task6.transform(data)

    X, y_test = sliding_window(data, 186)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    prediction = model_task6.predict(X)
    prediction = scaler_task6.inverse_transform(prediction)
    prediction = prediction.flatten().tolist()

    return {"prediction": prediction}
