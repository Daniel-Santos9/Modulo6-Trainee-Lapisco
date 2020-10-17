from django.shortcuts import render
from rest_framework import routers, serializers, viewsets
from .serializers import IrisSerializer
from .models import Iris
import numpy as np
# Create your views here.


#


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
