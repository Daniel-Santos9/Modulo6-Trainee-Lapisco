import joblib
from os import path
from .train import load_data       
import numpy as np                          

DIR_NAME = path.dirname(__file__)

MODELS_FOLDER = path.join(DIR_NAME, 'models')
EXPERIMENT_NAME = path.join(MODELS_FOLDER, 'exp_01_default')

from decouple import config as cfg
TRANSFORMER_NAME = cfg('TRANSFORMER_NAME', cast=str)
MODEL_NAME = cfg('MODEL_NAME', cast=str)

def load_models():
    tf = joblib.load(path.join(MODELS_FOLDER, EXPERIMENT_NAME,TRANSFORMER_NAME))
    model = joblib.load(path.join(MODELS_FOLDER, EXPERIMENT_NAME,MODEL_NAME))

    return model, tf

def check_inputs(input):
    print(input)

    if type(input) == list:
        if len(input) == 4:
            input = [float(i) for i in input]
            print(input)
            return np.array(input).reshape(1,-1)

    else:
        return 205

    pass