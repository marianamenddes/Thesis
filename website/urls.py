from django.urls import path, re_path
from . import views
from django.contrib.auth import views as auth_views
from django.contrib import admin

urlpatterns = [
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path("", views.Login, name="login"),
    path("details/<int:id>", views.Details, name="details"),
    path("details/patient/<int:id>", views.Patientinfo, name="pinfo"),
    path("details/patient/<int:id>/new_session", views.PatientSession, name="new_session"), 
    path("details/patient/<int:id>/session_<int:session_id>", views.SessionReview, name="sessionreview"),
    path('delete_session/<int:patient_pk>/<int:session_id>/', views.delete_session, name='delete_session'),
    path('get-data-points/<int:start>/<int:count>/<int:id>/<int:session_id>/', views.get_data_points, name='get_data_points'),
    path('get-expected-curve/<int:id>/', views.get_expected_curve, name='get_expected_curve'),
    path('get-data-between-indexes-right/<int:id>/<int:session_id>/<int:start_index>/<int:end_index>/', views.get_data_between_indexes_right, name='get_data_between_indexes_right'),
    path('get-data-between-indexes-left/<int:id>/<int:session_id>/<int:start_index>/<int:end_index>/', views.get_data_between_indexes_left, name='get_data_between_indexes_left'),
    path('get-all-data-points/<int:id>/<int:session_id>/', views.get_all_data_points, name='get_all_data_points'),
   
    path('get-emg-channel-data/<int:id>/<int:session_id>/<int:channel_num>/',
         views.get_emg_data_channel, name='get_emg_channel_data'),

    # 2. get_emg_data_between_indexes_channel (retorna UMA FATIA de dados de UM canal específico)
    path('get-emg-channel-slice/<int:id>/<int:session_id>/<int:channel_num>/<int:start_index>/<int:end_index>/',
         views.get_emg_data_between_indexes_channel, name='get_emg_channel_slice'),
]