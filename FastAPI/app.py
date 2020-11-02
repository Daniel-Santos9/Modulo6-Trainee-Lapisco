import argparse, joblib
from os import path

from train import load_data

from pydantic import BaseModel

from utils import load_models, check_inputs

from fastapi.encoders import jsonable_encoder

from fastapi import FastAPI
app = FastAPI()


class features_iris(BaseModel):
    sl: float
    sw: float
    pl: float
    pw: float

# Load models
model, tf = load_models()  

@app.get("/")
def inicio():
    return "Página Inicial, para utilizar o predict use /predict/"

@app.post("/predict/")
def predict(features: features_iris):
    f = [features.sl,features.sw, features.pl, features.pw]
    #features_dict = features_iris.dict()
    # Check inputs
    x = check_inputs(f)
    #tfm = tf.transform(x)
    print(tfm)
    y_hat = model.predict(x)

    yhat = str(y_hat)[1:-1]
    json_yhat = {"y_hat":yhat}
    return json_yhat





# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Iris classifier api 0.0.1')
#     parser.add_argument('--host', default='localhost', type=str)
#     parser.add_argument('--port', default=5000, type=str)
#     parser.add_argument('--debug', default=True, type=str)
#     args = vars(parser.parse_args()) 

#     # Load vars
#     model, tf = load_models()    

    # app.run(port=args['port'], debug=args['debug'])



