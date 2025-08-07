# C:\Users\maria\...\web_app\asgi.py

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web_app.settings')
django.setup()

from .routing import application