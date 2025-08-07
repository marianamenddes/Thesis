# C:\Users\maria\...\web_app\consumers.py

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.consumer import AsyncConsumer
from channels.db import database_sync_to_async
from django.utils import timezone
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from website.models import Sessions, Patient # Certifique-se de que a importação está correta

logger = logging.getLogger(__name__)

# Dicionários para armazenar dados temporariamente por ID de sessão
temp_session_data = {} 
active_sessions_status = {}
session_data_confirmed_by_django = {}

# Funções auxiliares (adaptadas da sua lógica original)
def get_session_status_from_consumer(session_db_id):
    return active_sessions_status.get(session_db_id, False)

def get_data_saved_ack_status(session_db_id):
    return session_data_confirmed_by_django.get(session_db_id, False)
@database_sync_to_async
def set_active_status_for_session(session_id, is_active):
    """
    Define o estado ativo de uma sessão na base de dados.
    """
    try:
        session = Sessions.objects.get(id=session_id)
        session.is_active = is_active
        session.save()
        logger.info(f"Status da sessão {session_id} atualizado para {is_active}.")
    except Sessions.DoesNotExist:
        logger.error(f"Erro: Sessão com o ID {session_id} não encontrada.")
    except Exception as e:
        logger.error(f"Ocorreu um erro ao atualizar o status da sessão: {e}")

@database_sync_to_async
def save_session_data(session_db_id):
    try:
        session_instance = Sessions.objects.get(id=session_db_id)
        session_data = temp_session_data.get(session_db_id, {})
        if session_instance.session_start_time:
            session_instance.session_time = (timezone.now() - session_instance.session_start_time).total_seconds()
        else:
            session_instance.session_time = 0 
        
        if session_data.get('session_type') in ["GAIT_ONLY", "COMBINED"]:
            session_instance.session_results_gait_right = ','.join(map(str, session_data['gait_r']))
            session_instance.session_results_gait_left = ','.join(map(str, session_data['gait_l']))
            session_instance.session_status_r = "Completed"
            session_instance.session_status_l = "Completed"

        if session_data.get('session_type') in ["EMG_ONLY", "COMBINED"]:
            session_instance.session_results_emg_channel1 = json.dumps(session_data['emg1'])
            session_instance.session_results_emg_channel2 = json.dumps(session_data['emg2'])
            session_instance.session_results_emg_channel3 = json.dumps(session_data['emg3'])
            session_instance.session_status_emg = "Completed"

        session_instance.save()
        logger.info(f"Dados da sessão {session_db_id} salvos no banco de dados.")

        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.send)(
            "mqtt.publish",
            {
                "action": "publish",
                "topic": "django/data_saved_ok", 
                "payload": str(session_db_id),
                "qos": 2,
            }
        )
        logger.info(f"Confirmação 'data_saved_ok' enviada para Arduino para sessão {session_db_id}.")
        
        # Limpar dados temporários APENAS AQUI, após o salvamento e a confirmação
        del temp_session_data[session_db_id]
        if session_db_id in active_sessions_status:
            del active_sessions_status[session_db_id]
        session_data_confirmed_by_django[session_db_id] = True 
        return True
    except Sessions.DoesNotExist:
        logger.error(f"Sessão {session_db_id} não encontrada para salvar dados.")
        return False
    except Exception as e:
        logger.error(f"Erro ao salvar dados da sessão {session_db_id} no DB: {e}")
        return False

# --- Consumidor para receber mensagens do Channel Layer (substitui o MyMqttConsumer) ---
class MqttReceiverConsumer(AsyncConsumer):
    async def mqtt_message_received(self, event):
        """
        Este método recebe as mensagens do run_mqtt_client.py através do Channel Layer.
        """
        payload_data = event['data']
        topic = payload_data.get('topic', '')
        payload_str = payload_data.get('payload', '')

        # Tenta extrair o ID da sessão e os dados
        try:
            payload_parts = payload_str.split('|', 1)
            if len(payload_parts) < 2:
                logger.warning("Payload inválido. Descartando.")
                return
            current_session_db_id = int(payload_parts[0])
            actual_payload = payload_parts[1]
        except (ValueError, IndexError):
            logger.error(f"Erro ao processar payload: {payload_str}")
            return

        if not active_sessions_status.get(current_session_db_id):
            logger.warning(f"Mensagem recebida para sessão inativa: {current_session_db_id}. Descartando.")
            return

        session_data = temp_session_data.get(current_session_db_id)
        if not session_data:
            logger.error(f"Dados temporários para sessão {current_session_db_id} não encontrados. Descartando.")
            return

        session_type = session_data['session_type']

        # Lógica para o comando 'End'
        if actual_payload == 'End':
            if topic == 'django/gait_values_right': session_data['gait_r_finished'] = True
            elif topic == 'django/gait_values_left': session_data['gait_l_finished'] = True
            elif topic.startswith('django/emg_values_channel'): session_data['emg_finished'] = True
            
            all_data_types_finished = False
            if session_type == "GAIT_ONLY": all_data_types_finished = session_data['gait_r_finished'] and session_data['gait_l_finished']
            elif session_type == "EMG_ONLY": all_data_types_finished = session_data['emg_finished']
            elif session_type == "COMBINED": all_data_types_finished = session_data['gait_r_finished'] and session_data['gait_l_finished'] and session_data['emg_finished']

            if all_data_types_finished:
                logger.info(f"Todos os tipos de dados da sessão {current_session_db_id} terminaram. Iniciando salvamento no DB.")
                await save_session_data(current_session_db_id)
            return
        
        # Lógica para dados numéricos
        try:
            values = [float(v) for v in actual_payload.split(',') if v.strip()]
            if topic.startswith('django/emg_values_channel'):
                values = [int(v) for v in values]
            
            if topic == 'django/gait_values_right' and session_type in ["GAIT_ONLY", "COMBINED"]:
                session_data['gait_r'].extend(values)
            elif topic == 'django/gait_values_left' and session_type in ["GAIT_ONLY", "COMBINED"]:
                session_data['gait_l'].extend(values)
            elif topic == 'django/emg_values_channel1' and session_type in ["EMG_ONLY", "COMBINED"]:
                session_data['emg1'].extend(values)
            elif topic == 'django/emg_values_channel2' and session_type in ["EMG_ONLY", "COMBINED"]:
                session_data['emg2'].extend(values)
            elif topic == 'django/emg_values_channel3' and session_type in ["EMG_ONLY", "COMBINED"]:
                session_data['emg3'].extend(values)

        except ValueError as e:
            logger.error(f"Erro ao parsear dados numéricos do tópico {topic}: {e} Payload: {actual_payload}")
        except Exception as e:
            logger.error(f"Erro inesperado no consumer para tópico {topic}: {e}")


# --- Consumer de WebSocket para a interface de utilizador ---
class PatientSessionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.patient_id = self.scope['url_route']['kwargs']['patient_id']
        self.room_group_name = f'patient_{self.patient_id}'
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        logger.info(f"WebSocket conectado para paciente {self.patient_id}")

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        logger.info(f"WebSocket desconectado para paciente {self.patient_id}")
    
    # Método para receber mensagens do Channel Layer e enviar para o WebSocket
    async def mqtt_message_to_websocket(self, event):
        payload_data = event['data']
        await self.send(text_data=json.dumps(payload_data))

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        action = text_data_json.get('action', '')
        if action == "begin_session_websocket" and 'session_info' in text_data_json:
            session_info = text_data_json['session_info']
            session_db_id = session_info.get('session_db_id')
            if session_db_id:
                # Inicializa os dados temporários
                temp_session_data[session_db_id] = {
                    'gait_r': [], 'gait_l': [], 'emg1': [], 'emg2': [], 'emg3': [],
                    'session_type': session_info.get('session_type'),
                    'emg_leg_choice': session_info.get('emg_leg_choice'),
                    'gait_r_finished': False, 'gait_l_finished': False, 'emg_finished': False,
                }
                active_sessions_status[session_db_id] = True
                logger.info(f"Sessão {session_db_id} iniciada via WebSocket. Dados temporários limpos.")
                # Envia o comando de início ao Arduino via MQTT
                channel_layer = get_channel_layer()
                await channel_layer.send(
                    "mqtt.publish",
                    {
                        "action": "publish",
                        "topic": "django/session_start_info",
                        "payload": json.dumps(session_info),
                        "qos": 2,
                    }
                )