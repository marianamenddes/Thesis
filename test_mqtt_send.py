# test_mqtt_send.py

import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_app.settings")
django.setup()


from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

channel_layer = get_channel_layer()

async_to_sync(channel_layer.send)(
    "mqtt.publish",
    {
        "type": "mqtt_publish",
        "action": "publish",
        "topic": "test/topic",
        "payload": "Hello!",
        "qos": 1,
    }
)
