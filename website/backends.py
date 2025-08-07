from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.hashers import check_password
from .models import Doctor

class DoctorBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            user = Doctor.objects.get(name=username)
            if check_password(password, user.password):
                return user
        except Doctor.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return Doctor.objects.get(pk=user_id)
        except Doctor.DoesNotExist:
            return None