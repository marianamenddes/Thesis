# C:\Users\maria\...\web_app\run_mqtt_client.py

# Configura o Django Settings
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web_app.settings') # Substitua 'myproject' pelo nome do seu projeto
django.setup()

import paho.mqtt.client as mqtt
import json
import logging
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.conf import settings


logger = logging.getLogger(__name__)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Conectado ao broker MQTT com sucesso.")
        # Subscrever aos tópicos relevantes
        client.subscribe('django/gait_values_right')
        client.subscribe('django/gait_values_left')
        client.subscribe('django/emg_values_channel1')
        client.subscribe('django/emg_values_channel2')
        client.subscribe('django/emg_values_channel3')
        logger.info("Subscrito a todos os tópicos de dados do Arduino.")
    else:
        logger.error(f"Falha na conexão, código de erro: {rc}")

def on_message(client, userdata, msg):
    try:
        payload_str = msg.payload.decode('utf-8')
        logger.debug(f"Mensagem recebida no tópico '{msg.topic}' com payload: '{payload_str}'")

        # Envia a mensagem para o Channel Layer
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.send)(
            "mqtt.receive",
            {
                "type": "mqtt.message.received",
                "data": {
                    "topic": msg.topic,
                    "payload": payload_str,
                }
            }
        )
    except Exception as e:
        logger.error(f"Erro ao processar mensagem MQTT: {e}")

# Configurações do broker
BROKER = settings.MQTT_BROKER
PORT = settings.MQTT_PORT

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER, PORT)
    client.loop_forever()
except Exception as e:
    logger.error(f"Erro ao conectar ao broker MQTT: {e}")