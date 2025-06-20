from django.urls import path
from .views import index, flow_api

urlpatterns = [
    path('', index, name='index'),
    path('api/flow', flow_api, name='flow_api'),
] 