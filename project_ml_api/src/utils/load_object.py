import pickle
from tensorflow.keras.models import load_model


def load_model_object(name: str):
    return load_model(f"src/objects/{name}")


def load_scaler_object(name: str):
    return pickle.load(open(f"src/objects/{name}", 'rb'))
