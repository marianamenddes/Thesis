import json
import time
import paho.mqtt.client as mqtt
from channels.consumer import SyncConsumer
from django.conf import settings

# --- MQTT SETUP ---
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker for publishing.")
    else:
        print(f"❌ Failed to connect to MQTT broker. Code: {rc}")

def on_disconnect(client, userdata, rc):
    print("⚠️ MQTT disconnected with code:", rc)
    print("🔁 Trying to reconnect...")
    time.sleep(5)
    try:
        client.reconnect()
    except Exception as e:
        print("❌ Reconnection failed:", e)

def connect_mqtt():
    try:
        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print("❌ MQTT connection error:", e)

# --- CHANNEL CONSUMER ---
class MqttPublisherConsumer(SyncConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("🔌 MqttPublisherConsumer initialized")
        connect_mqtt()

    def mqtt_publish(self, message):
        print("Mensagem recebida:", message)

        try:
            action = message.get("action")
            topic = message.get("topic")
            payload = message.get("payload")
            qos = message.get("qos", 0)

            if action == "publish" and topic and payload is not None:
                mqtt_client.publish(topic, payload, qos)
                print(f"✅ Published to '{topic}': {payload}")
            else:
                print("❗Mensagem ignorada (faltam campos)")
        except Exception as e:
            print(f"❌ Erro ao publicar no MQTT: {e}")
