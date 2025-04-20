from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('summarize/', views.summarize_view, name='summarize'),
    path('sentiment/', views.sentiment_view, name='sentiment'),
    path('upload/', views.upload_file_view, name='upload'),
    path('export-pdf/', views.export_pdf_view, name='export_pdf'),
]
