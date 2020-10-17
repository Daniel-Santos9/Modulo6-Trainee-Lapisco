from django.urls import path, include
#from django.contrib.auth.models import Iris
from rest_framework import routers, serializers, viewsets
from .models import Iris
# Serializers define the API representation.
class IrisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Iris
        fields = ['features', 'label']