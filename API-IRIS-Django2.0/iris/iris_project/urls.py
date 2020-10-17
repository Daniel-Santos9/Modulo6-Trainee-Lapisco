from django.urls import path
from .views import predict, predict_test



urlpatterns = [
    path('', predict_test, name='predict_test'),
    path('predict/', predict, name='predict'),
]