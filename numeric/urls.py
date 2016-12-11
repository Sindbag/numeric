from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^auto/$', views.auto, name='auto'),
    url(r'^$', views.manual, name='manual'),
]