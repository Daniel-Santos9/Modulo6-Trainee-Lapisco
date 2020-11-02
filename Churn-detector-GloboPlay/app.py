import joblib
import argparse
from os import path
import numpy as np
from train import load_data
from utils import load_models, check_inputs
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

class Item(BaseModel):
    features: list
    label: Optional[str] = None

app = FastAPI()

model, tf = load_models()

@app.post("/predict/")
def predict(item: Item):
    print(item.features)
    #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: ", len(item.features))
    x = check_inputs(item.features)
    x_tf = tf.transform(x)
    print("passou do transform")
    y_hat = model.predict(x_tf)
    print("y_hat", y_hat[0])
    if(y_hat[0] == 0):
        pred = "Assinante"
    else:
        pred = "Cancelou"
    item.label = pred

    return item

@app.get("/")
def read_root():
    return {"Hello": "FastAPI"}
    


