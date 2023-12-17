import pickle
from tensorflow.keras.models import load_model


def load_models_task4():
    model = pickle.load(open('app/data/task4/regressor.pkl', 'rb'))
    scaler = pickle.load(open('app/data/task4/scaler.pkl', 'rb'))
    return model, scaler


def load_models_task6():
    model = load_model('app/data/task6/model_gru.h5')
    scaler = pickle.load(open('app/data/task6/scaler.pkl', 'rb'))
    return model, scaler


def load_models_task5():
    model = load_model('app/data/task5/classifier.h5')
    return model
