from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from api import views

urlpatterns = [
    path('classify/',views.Classify.as_view()),
]

urlpatterns = format_suffix_patterns(urlpatterns)