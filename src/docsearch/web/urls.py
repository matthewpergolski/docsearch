from django.urls import path, re_path

from . import views
from .api import api

urlpatterns = [
    path("api/", api.urls),
    path("", views.index),
    re_path(r"^assets/(?P<path>.+)$", views.asset),
]
