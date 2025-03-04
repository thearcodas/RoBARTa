from django.contrib import admin
from django.urls import path,re_path
from dashboard import views
from django.conf import settings


urlpatterns = [
    path('', views.dashboard,name='dashboard'),]