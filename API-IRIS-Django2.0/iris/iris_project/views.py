from django.shortcuts import render

from django.shortcuts import render
from .models import Iris

from .train import load_data
from .utils import load_models, check_inputs
from django.http import HttpResponse, JsonResponse
from .forms import IrisForm
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import argparse
from os import path
import numpy as np
import sys

sys.path.append('../iris')

model, tf = load_models()


@csrf_exempt
def predict(request):

    df = json.loads(request.body)

    x = list((df['features']).split(","))
    x = np.array(x).reshape(1,-1)
    y_hat = model.predict(tf.transform(x))
    print("------------------------------------------")
    print(y_hat)
    print("------------------------------------------")
    df['label'] = str(y_hat)

    return JsonResponse(df)



@csrf_exempt
def predict_test(request):
    X, y = load_data()
    y_hat = model.predict(tf.transform(X))
    df = {'predict_test': 'predict'}

    return JsonResponse(df)


