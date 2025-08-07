# C:\Users\maria\...\web_app\routing.py

from django.urls import re_path
from channels.routing import ProtocolTypeRouter, URLRouter, ChannelNameRouter
from . import consumers, mqtt_publisher
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack

websocket_urlpatterns = [
    re_path(r'ws/patientsession/(?P<patient_id>\d+)/$', consumers.PatientSessionConsumer.as_asgi()),
]

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
    "channel": ChannelNameRouter({
        "mqtt.publish": mqtt_publisher.MqttPublisherConsumer.as_asgi(),
        "mqtt.receive": consumers.MqttReceiverConsumer.as_asgi(),
    }),
})