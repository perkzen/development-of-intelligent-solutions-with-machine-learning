from pydantic import BaseModel


class AirQuality(BaseModel):
    pm10: float
    pm25: float
    pm25_o3: float
    pm25_no2: float

