from pydantic import BaseModel


class WeatherData(BaseModel):
    temperature: float
    apparent_temperature: float
    precipitation_probability: float
    hour: int
    apparent_temperature_difference: float


class ExtendedWeatherData(BaseModel):
    date: str
    temperature: float
    relative_humidity: float
    dew_point: float
    apparent_temperature: float
    precipitation_probability: float
    rain: float
    surface_pressure: float
    bike_stands: int
    available_bike_stands: int
