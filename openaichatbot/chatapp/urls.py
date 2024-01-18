from django.urls import path
from . import views

urlpatterns = [
    path("index/",views.index,name='indexpage'),
    
    path('', views.chatbot, name='chatbot'),
    
]
