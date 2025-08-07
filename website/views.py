from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from .models import Doctor, Patient, Sessions
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
import calendar
from calendar import HTMLCalendar, monthrange
from datetime import datetime, timedelta
import torch
import paho.mqtt.client as paho
from django.contrib import messages
import os
import pandas as pd
import tensorflow as tf
from django.utils import timezone
import joblib
from django.conf import settings
from .models_ml.knee_model import KneeFlexionModel
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .functions import preprocess_emg_signal
import json
import numpy as np
from .utilities.plot import generate_plot, generate_expected_curve
from scipy.signal import savgol_filter
from .form import PatientForm
import logging
import joblib
import xgboost as xgb
from django.utils import timezone
from .backends import DoctorBackend
from django.contrib.auth.models import User
import random
from django.contrib.admin.views.decorators import staff_member_required
from datetime import date # NECESSÁRIO para comparar com o dia de hoje
from django.http import JsonResponse # NECESSÁRIO para a resposta AJAX do formulário
import random # NECESSÁRIO para o número do paciente
import logging # NECESSÁRIO para os logs de erro
from django.shortcuts import render, redirect, get_object_or_404 # Garante que redirect e get_object_or_404 estão lá
from django.utils import timezone # Já a deves ter, mas confirma

# Consumers e channel layer (Permanecem aqui para interagir com o consumer)
from web_app import consumers # Importa o seu MyMqttConsumer e funções auxiliares


class CustomHTMLCalendar(calendar.HTMLCalendar): # Renomeei para CustomHTMLCalendar para evitar conflito
    def formatday(self, day, weekday):
        """
        Retorna uma célula de tabela (<td>) para um dia.
        Adiciona a classe 'current-day' ao dia de hoje.
        """
        if day == 0:
            return '<td class="noday">&nbsp;</td>'  # Dia fora do mês
        else:
            cssclass = self.cssclasses[weekday]
            # Verifica se é o dia, mês e ano atuais
            if date.today().day == day and \
               date.today().month == self.month and \
               date.today().year == self.year:
                cssclass += ' current-day' # Adiciona a classe 'current-day'
            return '<td class="%s">%d</td>' % (cssclass, day)

    def formatmonth(self, theyear, themonth, withyear=True):
        """
        Retorna um mês como uma tabela HTML.
        Armazena o ano e o mês para uso em formatday.
        """
        self.year = theyear
        self.month = themonth
        return super().formatmonth(theyear, themonth, withyear=withyear)
        
logger = logging.getLogger(__name__)
# Permitir múltiplas versões do OpenMP (não recomendado para produção)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Limitar o número de threads
os.environ["OMP_NUM_THREADS"] = "1"

# Create your views here.
BASE_DIR = settings.BASE_DIR
model_path = os.path.join(BASE_DIR, 'website', 'models_ml', 'KneeFlexionModel.pth')
# Verifica se o arquivo do modelo existe antes de tentar carregar
if os.path.exists(model_path):
    model = KneeFlexionModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Colocar o modelo em modo de avaliação
else:
    logging.error(f"KneeFlexionModel.pth não encontrado em: {model_path}")
    model = None # Ou lide com isso de outra forma, como levantar um erro ou usar um modelo dummy


# --- CARREGAMENTO DOS DOIS MODELOS EMG (Keras/TensorFlow) ---
# Caminhos para o modelo e scalers DOMINANTE
EMG_DOM_MODEL_H5_PATH = os.path.join(BASE_DIR, 'website', 'models_ml', 'marionet_emg_model_dominant.h5')
EMG_DOM_MEAN_PKL_PATH = os.path.join(BASE_DIR, 'website', 'models_ml', 'marionet_emg_mean_dominant.pkl')
EMG_DOM_STD_PKL_PATH = os.path.join(BASE_DIR, 'website', 'models_ml', 'marionet_emg_std_dominant.pkl')

# Caminhos para o modelo e scalers NÃO-DOMINANTE
EMG_NDOM_MODEL_H5_PATH = os.path.join(BASE_DIR, 'website', 'models_ml', 'marionet_emg_model_non_dominant.h5')
EMG_NDOM_MEAN_PKL_PATH = os.path.join(BASE_DIR, 'website', 'models_ml', 'marionet_emg_mean_non_dominant.pkl')
EMG_NDOM_STD_PKL_PATH = os.path.join(BASE_DIR, 'website', 'models_ml', 'marionet_emg_std_non_dominant.pkl')

loaded_emg_dom_model, loaded_emg_dom_mean, loaded_emg_dom_std = None, None, None
loaded_emg_ndom_model, loaded_emg_ndom_mean, loaded_emg_ndom_std = None, None, None
# Já temos um logger configurado acima, não precisamos de outro aqui.
# logger = logging.getLogger(__name__) # Removido
#carregamento do modelos dos joelhos
model_gait_path = os.path.join(BASE_DIR, 'website', 'models_ml', "rf_koa_model.pkl")
scaler_gait_path = os.path.join(BASE_DIR, 'website', 'models_ml',"scaler_rf_koa.pkl")
model_severity_path = os.path.join(BASE_DIR, 'website', 'models_ml',"xgb_severity_model.pkl")
model_gait, scaler_gait, model_severity = None, None, None

try:
    loaded_emg_dom_model = tf.keras.models.load_model(EMG_DOM_MODEL_H5_PATH)
    loaded_emg_dom_mean = joblib.load(EMG_DOM_MEAN_PKL_PATH)
    loaded_emg_dom_std = joblib.load(EMG_DOM_STD_PKL_PATH)
    logger.info("Modelo EMG Dominante e scalers carregados com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar modelo EMG Dominante ou scalers: {e}")

try:
    loaded_emg_ndom_model = tf.keras.models.load_model(EMG_NDOM_MODEL_H5_PATH)
    loaded_emg_ndom_mean = joblib.load(EMG_NDOM_MEAN_PKL_PATH)
    loaded_emg_ndom_std = joblib.load(EMG_NDOM_STD_PKL_PATH)
    logger.info("Modelo EMG Não-Dominante e scalers carregados com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar modelo EMG Não-Dominante ou scalers: {e}")

try:
    model_gait = joblib.load(model_gait_path)
    scaler_gait = joblib.load(scaler_gait_path)
except Exception as e:
    print(f"Erro ao carregar modelo/scaler gait: {e}")

try:
    model_severity = xgb.Booster()
    model_severity.load_model(model_severity_path)
except Exception as e:
    print

def custom_admin_page(request):
    return render(request, 'admin/custom_admin_page.html')

def Login(request):
    if request.method == "POST":
        # Cria a instância do formulário, passando os dados do POST
        form = AuthenticationForm(request, data=request.POST) 
        
        # Verifica se o formulário é válido (o Django lida com campos vazios, etc.)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')

            # Tenta autenticar o utilizador usando o teu backend personalizado
            user = DoctorBackend().authenticate(request, username=username, password=password)
            
            if user is not None:
                # Login bem-sucedido
                login(request, user, backend='website.backends.DoctorBackend')
                print(f"User {user.name} authenticated successfully.")
                return redirect("details/" + str(user.id)) # Redireciona após login
            else:
                # Autenticação falhou (credenciais inválidas)
                # Adiciona um erro ao formulário para que possa ser exibido no template
                form.add_error(None, "Nome de utilizador ou palavra-passe inválidos.") 
                # E continua para a parte de renderizar o template com o formulário e erros
        
        # Se o método for POST e o formulário NÃO for válido (ou autenticação falhou no 'else' acima)
        # Renderiza o template de login novamente, passando o formulário (que agora contém erros)
        return render(request, "loginpage.html", {"form": form}) 

    else: # Isto é para um pedido GET (quando a página é carregada pela primeira vez)
        form = AuthenticationForm() # Cria uma instância vazia do formulário
    
    # Renderiza o template de login, passando sempre a instância do formulário
    return render(request, "loginpage.html", {"form": form})



def Details(request, id):
    try:
        user = request.user
        doc = user
        patients = Patient.objects.filter(doc=doc)

        # --- ALTERAÇÃO AQUI: USAR A TUA CLASSE CUSTOMIZADA ---
        cal_instance = CustomHTMLCalendar() # Usar CustomHTMLCalendar
        current_date = timezone.now()
        calendar_html = cal_instance.formatmonth(current_date.year, current_date.month).replace(
             '<td ', '<td width="50" height="50"')
        # ----------------------------------------------------

        doc.last_login = timezone.now()
        doc.save()
        logger.debug(f"Details view acessada por User ID: {user.id}, Doctor ID: {doc.id}") # Adicione esta linha

        if request.method == 'POST':
            form = PatientForm(request.POST)
            if form.is_valid():
                patient = form.save(commit=False)
                patient.doc = doc
                patient.patient_number = random.randint(1, 9999999)
                patient.save()
                logger.info(f"Paciente '{patient.name}' (ID: {patient.id}, Número: {patient.patient_number}) guardado para Doutor ID: {doc.id}")
                return JsonResponse({'success': True}) # Resposta para AJAX
            else:
                logger.warning(f"Formulário de Paciente inválido para User ID: {user.id}. Erros: {form.errors.as_json()}") # Adicione esta linha
                return JsonResponse({'success': False, 'errors': form.errors}) # Erros para AJAX
        else:
            form = PatientForm()

        
        logger.debug(f"A buscar pacientes para Doutor ID: {doc.id}. Encontrados {patients.count()} pacientes.") # Adicione esta linha

        context = {
            "doc": doc,
            "patients": patients,
            "calendar": calendar_html, # A variável no contexto é "calendar"
            "form": form,
            "user": request.user # Passar o user para o template
        }
        return render(request, "details.html", context)
    except Doctor.DoesNotExist:
        logger.error(f"Doctor with id {id} does not exist.")
        return render(request, "error.html", {'message': 'Doctor not found.'}, status=404)
    except Exception as e:
        logger.error(f"Error loading details: {e}")
        return render(request, "error.html", {'message': 'An unexpected error occurred.'})

def Patientinfo(request, id):
    patient = get_object_or_404(Patient, id=id) # Usar get_object_or_404
    doc = get_object_or_404(Doctor, id=patient.doc_id) # Usar get_object_or_404
    all_sessions = Sessions.objects.filter(Patient_id=id).order_by('-date') # Renomeado 'session' para 'sessions' para evitar conflito

    recent_sessions = all_sessions[:3]
    other_sessions = all_sessions[3:]

    # Lógica para a curva esperada do joelho, apenas se o modelo estiver carregado
    smoothed_curve = []
    if model:
        sex = 1 if patient.sex == 'Male' else 0
        with torch.no_grad():
            knee_angle_curve = model(torch.tensor([[patient.age, sex, patient.weight, patient.height]]))
        smoothed_curve = savgol_filter(knee_angle_curve[0].cpu().detach().numpy(), 15, 4).tolist()
    
    last_emg_prediction_text = "N/A"
    if all_sessions.exists(): # Usar 'sessions' aqui
        # Tenta encontrar a sessão mais recente que é de EMG ou Combinada e tem uma predição
        last_session_with_emg_pred = all_sessions.filter(session_type__in=['EMG_ONLY', 'COMBINED']).exclude(emg_prediction__isnull=True).exclude(emg_prediction__exact='').order_by('-session_start_time').first() # Adicionado order_by
        if last_session_with_emg_pred:
            last_emg_prediction_text = last_session_with_emg_pred.emg_prediction
        else:
            last_emg_prediction_text = "Nenhuma predição EMG disponível para sessões de EMG/Combinadas."


    if request.method == 'POST':
        if 'go_home' in request.POST:
            return redirect('/details/' + str(doc.id)) # Redireciona para a página de detalhes do doutor

    context = {
        "patient": patient,
        "recent_sessions": recent_sessions,  # Passa as 3 mais recentes
        "other_sessions": other_sessions,    # Passa as restantes (para o "mostrar mais")
        "has_more_sessions": all_sessions.count() > 3,
        "expected_knee_curve": smoothed_curve, 
        "last_emg_prediction_text": last_emg_prediction_text,
    }
    print("DEBUG: Carregando views.py para Patientinfo") # Mudei para Patientinfo
    return render(request, "patientinfo.html", context)


# --- INÍCIO DA FUNÇÃO PatientSession ---
def PatientSession(request, id):
    patient = get_object_or_404(Patient, id=id)
    session_status_message = None 
    notification = None
    message = None
    notification_pop = None
    new_exam_start_message = "" # Inicializar a mensagem aqui

    channel_layer = get_channel_layer() 
    
    last_session_db_id = None
    latest_session = Sessions.objects.filter(Patient=patient).order_by('-session_start_time').first()

    is_any_session_active_in_consumer = False # Esta variável pode ser removida se não for usada noutro sítio

    try:
        if hasattr(consumers, 'active_sessions_status'):
            # Verifica se a última sessão deste paciente está 'Ongoing' E 'ativa' no consumidor
            if latest_session and \
               latest_session.id in consumers.active_sessions_status and \
               consumers.active_sessions_status.get(latest_session.id) and \
               latest_session.session_status_r == "Ongoing": # Assumindo que session_status_r reflete o estado geral
                logger.info(f"Session {latest_session.id} is ongoing (DB) and active (consumer). Redirecting to SessionReview.")
                return redirect('sessionreview', id=patient.id, session_id=latest_session.id)
            else:
                # Se a última sessão está "Completed" mas ainda aparece como ativa no consumidor, limpa o status
                if latest_session and \
                   latest_session.id in consumers.active_sessions_status and \
                   consumers.active_sessions_status.get(latest_session.id) and \
                   latest_session.session_status_r == "Completed": # Ou qualquer outro status 'terminado'
                    async_to_sync(consumers.set_active_status_for_session)(latest_session.id, False)
                    logger.info(f"Stale active status for completed session {latest_session.id} cleared.")

    except NameError:
        logger.warning("Consumer module or active_sessions_status not found. Session activity check skipped.")
        # Se o módulo consumer não for encontrado, is_any_session_active_in_consumer deve ser False

    # Variables for dynamic page content (initial values)
    session_type_display_name = "Session Setup" 
    video_url = ""
    explanation_text = "Please select a session type to view detailed instructions." 
    diagram_image = "" 
    preparation_instructions = "" 
    show_start_collection_button = False 
    show_session_actions_menu = False 

    selected_session_type = None
    emg_leg_choice = 'Right' # Definir um valor padrão para emg_leg_choice
    collection_duration_minutes = 5


    # --- Lógica de Carregamento da Sessão Existente ou Processamento do POST ---

    # Se a página é carregada via GET ou se a sessão mais recente já existe
    if latest_session:
        selected_session_type = latest_session.session_type
        last_session_db_id = latest_session.id
        emg_leg_choice = latest_session.emg_measured_leg if latest_session.emg_measured_leg else 'Right' # Pega a perna salva no DB ou default
        collection_duration_minutes = latest_session.collection_duration_minutes if latest_session.collection_duration_minutes else 5 # Pega a duração salva ou default

        # Esta mensagem será definida novamente abaixo no bloco 'if selected_session_type'
        # session_status_message = f"Exam type chosen: {selected_session_type.replace('_', ' ')}. Duration: {collection_duration_minutes} min (adjustable). Press 'Start Exam' to proceed."
        show_start_collection_button = True
    
    # Processa requisições POST
    if request.method == "POST":
        # Processa a atualização da duração (se vier no POST do Start Exam)
        try:
            collection_duration_minutes = int(request.POST.get('collection_duration_minutes', collection_duration_minutes))
            if collection_duration_minutes <= 0: 
                collection_duration_minutes = 5
        except (ValueError, TypeError):
            collection_duration_minutes = 5 

        # --- Lida com a seleção do TIPO DE SESSÃO (Gait, EMG, Combined) ---
        # Estas são as ações para criar uma NOVA sessão ou mudar o tipo de sessão para uma PENDING
        if 'session_gait_selected' in request.POST or \
           'session_emg_selected' in request.POST or \
           'session_combined_selected' in request.POST:
            
            if 'session_gait_selected' in request.POST:
                selected_session_type = "GAIT_ONLY"
            elif 'session_emg_selected' in request.POST:
                selected_session_type = "EMG_ONLY"
            elif 'session_combined_selected' in request.POST:
                selected_session_type = "COMBINED"

            # O emg_leg_choice para a criação da sessão será 'Right' por padrão ou o que já estiver no latest_session
            # Ele será atualizado DEPOIS pela caixa de seleção de perna separada.

            # Cria ou atualiza a sessão no DB
            if not latest_session or (latest_session.session_type != selected_session_type and not is_any_session_active_in_consumer):
                num = patient.session_num + 1 if patient.session_num else 1
                patient.session_num = num
                patient.save()

                session = Sessions.objects.create(
                    Patient=patient,
                    session_id=num,
                    session_type=selected_session_type,
                    session_status_r="Pending", 
                    session_status_l="Pending",
                    session_status_emg="Pending",
                    emg_measured_leg=emg_leg_choice, # Usa o valor padrão ou o que já existia
                    session_start_time=timezone.now(),
                    collection_duration_minutes=collection_duration_minutes
                )
                session.save()
                latest_session = session 
                last_session_db_id = session.id
                logger.info(f"DB Session ID {session.id} created (type: {selected_session_type}) for patient {patient.id}. Awaiting 'Start Collection'.")
                # A notificação será definida após a população do conteúdo dinâmico
            elif latest_session and latest_session.session_type == selected_session_type:
                notification = f"Session type already set to {selected_session_type.replace('_', ' ')}."
            
            # Atualiza selected_session_type e emg_leg_choice para o contexto, caso a página re-renderize
            if latest_session:
                selected_session_type = latest_session.session_type
                emg_leg_choice = latest_session.emg_measured_leg if latest_session.emg_measured_leg else 'Right'
                collection_duration_minutes = latest_session.collection_duration_minutes if latest_session.collection_duration_minutes else 5


        # --- Lida APENAS com a atualização da escolha da perna (NOVO BLOCO do último ajuste) ---
        if 'update_emg_leg_choice' in request.POST:
            new_emg_leg = request.POST.get('emg_measured_leg')
            if latest_session and new_emg_leg:
                latest_session.emg_measured_leg = new_emg_leg
                latest_session.save()
                emg_leg_choice = new_emg_leg # Atualiza a variável para re-renderizar
                notification = f"EMG Measured Leg updated to {new_emg_leg}."
                logger.info(f"EMG Measured Leg updated to {new_emg_leg} for session {latest_session.id}.")
            else:
                notification = "Could not update EMG Measured Leg. Session not found or no leg selected."
            
            # Se for apenas uma atualização de perna, re-renderiza a página com o estado atual
            # A página já estará carregada com o latest_session correto
            # selected_session_type e collection_duration_minutes já estarão definidos a partir do latest_session
            # Não precisamos de redirect aqui, apenas deixamos a função renderizar o contexto atualizado.


        # Lógica para "Start Collection" button (mantém o mesmo comportamento)
        if 'start_session_confirm' in request.POST:
            if latest_session: 
                last_session_db_id = latest_session.id
                
                # Garante que a duração mais recente é salva
                #latest_session.collection_duration_minutes = collection_duration_minutes
                #latest_session.save()

                session_info_payload = {
                    "session_db_id": last_session_db_id,
                    "session_type": latest_session.session_type,
                    "emg_leg_choice": latest_session.emg_measured_leg,
                    "collection_duration_minutes": latest_session.collection_duration_minutes
                }
                channel_layer = get_channel_layer()
                async_to_sync(channel_layer.send)(
                    "mqtt.publish",
                    {
                        "type": "mqtt_publish",
                        "action": "publish",
                        "topic": "django/session_start_info",
                        "payload": f"{last_session_db_id}", #json.dumps(session_info_payload),
                        "qos": 2,
                    }
                )
                logger.info(f"DB Session ID {last_session_db_id} confirmed. Info sent to Arduino.")

                async_to_sync(channel_layer.send)(
                    "mqtt.publish",
                    {
                        "type": "mqtt_publish",
                        "action": "publish",
                        "topic": "django/confirm",
                        "payload": "Begin",
                        "qos": 2,
                    }
                )
                logger.info(f"Command 'Begin' sent to Arduino for session {last_session_db_id}.")
                
                latest_session.session_status_r = "Ongoing"
                latest_session.session_status_l = "Ongoing"
                latest_session.session_status_emg = "Ongoing"
                latest_session.save()
                
                try:
                    if hasattr(consumers, 'active_sessions_status'):
                        consumers.active_sessions_status[last_session_db_id] = True
                        is_any_session_active_in_consumer = True
                except NameError:
                    pass
                
                notification = "Data collection initiated! Please proceed to the review page."
                session_status_message = "Session ongoing. Awaiting data from Arduino..."
                return redirect('sessionreview', id=patient.id, session_id=last_session_db_id)
            else:
                notification = "No session configured to start. Please go back and select a session type."
                show_start_collection_button = False

        elif 'go_back_to_list' in request.POST: # Este botão está no menu hamburguer
            logger.info(f"Returning to patient details for patient {patient.id}.")
            return redirect('details', id=patient.doc_id) 

    # --- Popula conteúdo dinâmico baseado no tipo de sessão atual (vindo do DB ou de um POST) ---
    if selected_session_type:
        if selected_session_type == "EMG_ONLY":
            session_type_display_name = "Electromyography Session"
            explanation_text = (
                "This session focuses on the detailed analysis of electromyographic (EMG) muscle activity. "
                "The patient will need to follow specific guidelines for electrode placement, as illustrated in the accompanying diagram. "
                "EMG data will be meticulously recorded for subsequent biomechanical and clinical assessment. "
                "Ensure patient comfort and optimal electrode contact throughout the procedure. The video below demonstrates a typical EMG session setup."
            )
            preparation_instructions = (
                "**Electrode Placement Protocol:**\n"
                "1.  **Skin Preparation:** Prior to electrode application, the skin surface should be thoroughly cleaned with an alcohol swab to remove any oils or debris. For areas with dense hair, gentle shaving or trimming may be necessary to ensure optimal skin-electrode impedance.\n"
                "2.  **Electrode Application:** Refer to the anatomical diagrams (Image: EMG Electrode Placement Diagram) for precise electrode positioning over the target muscles. Ensure firm and even contact of the electrodes with the skin.\n"
                "3.  **Cable Management:** Secure electrode cables to minimize movement artifacts during patient activity.\n"
                "4.  **Patient Comfort:** Verify that the patient is comfortable and understands the instructions. Any discomfort should be addressed before proceeding."
            )
            
            # --- Lógica para VÍDEO e DIAGRAMA de EMG por Perna ---
            if emg_leg_choice == "Right":
                video_url = "http://googleusercontent.com/youtube.com/emg_right_leg_video" # SEU VÍDEO EMG PERNA DIREITA
                diagram_image = "img/emg_right_leg_diagram.png" # SUA IMAGEM EMG PERNA DIREITA
            elif emg_leg_choice == "Left":
                video_url = "http://googleusercontent.com/youtube.com/emg_left_leg_video" # SEU VÍDEO EMG PERNA ESQUERDA
                diagram_image = "img/emg_left_leg_diagram.png" # SUA IMAGEM EMG PERNA ESQUERDA
            else: # Default ou se não houver perna selecionada (improvável com o padrão 'Right')
                video_url = "http://googleusercontent.com/youtube.com/emg_default_video" 
                diagram_image = "img/emg_default_diagram.png" 

            show_start_collection_button = True
        
        elif selected_session_type == "GAIT_ONLY":
            session_type_display_name = "Gait Analysis Session"
            explanation_text = (
                "This session involves a comprehensive analysis of the patient's gait pattern. "
                "The patient will be instructed to walk across a designated pathway at their natural and comfortable pace. "
                "Kinematic and kinetic data, including stride length, cadence, velocity, and joint angles, will be captured. "
                "Ensure the patient has sufficient clear space to ambulate freely. The video below illustrates a standard gait analysis procedure."
            )
            preparation_instructions = (
                "**Knee Brace/Sensor Placement Protocol:**\n"
                "1.  **Device Preparation:** Ensure the knee brace/gait sensors are fully charged and calibrated according to manufacturer guidelines.\n"
                "2.  **Sensor Attachment:** Carefully attach the gait sensors to the designated anatomical landmarks, as indicated in the diagram (Image: Gait Sensor Placement Diagram). Verify secure attachment to prevent slippage during movement.\n"
                "3.  **Patient Posture:** Instruct the patient to adopt a natural standing posture. Ensure footwear is appropriate and consistent with their daily activities.\n"
                "4.  **Trial Walk:** Conduct a brief trial walk to confirm sensor stability and data integrity before initiating the formal collection."
            )
            video_url = "http://googleusercontent.com/youtube.com/gait_video" # SEU VÍDEO GAIT
            diagram_image = "img/gait_diagram.png" # SUA IMAGEM GAIT
            show_start_collection_button = True
        
        elif selected_session_type == "COMBINED":
            session_type_display_name = "Combined EMG & Gait Session"
            explanation_text = (
                "This integrated session simultaneously captures Electromyography (EMG) data alongside comprehensive Gait Analysis. "
                "The patient will perform ambulation tasks while concurrent EMG activity is recorded. "
                "This synergistic approach provides a holistic understanding of neuromuscular function during dynamic locomotion. "
                "Meticulous setup of both EMG electrodes and gait sensors is paramount for data synchronization and accuracy. The video below presents an overview of a combined session setup."
            )
            preparation_instructions = (
                "**Combined Sensor Placement Protocol:**\n"
                "1.  **EMG Electrode Application:** Follow the detailed steps for skin preparation and electrode placement as described for the EMG-only session. Refer to the EMG electrode diagram for precise positioning.\n"
                "2.  **Knee Brace/Gait Sensor Attachment:** Securely attach gait sensors to the designated anatomical landmarks as per the Gait Analysis protocol, consulting the respective diagram.\n"
                "3.  **Integrated Check:** Perform a final visual inspection to ensure all sensors and electrodes are properly affixed and that cables are managed to prevent interference.\n"
                "4.  **Patient Orientation:** Brief the patient on the combined nature of the session and ensure they are comfortable with both sets of devices. Confirm their understanding of the tasks to be performed. The video below showcases the combined setup and initial procedures."
            )
            
            # --- Lógica para VÍDEO e DIAGRAMA de COMBINED por Perna ---
            if emg_leg_choice == "Right":
                video_url = "http://googleusercontent.com/youtube.com/combined_right_leg_video" # SEU VÍDEO COMBINED PERNA DIREITA
                diagram_image = "img/combined_right_leg_diagram.png" # SUA IMAGEM COMBINED PERNA DIREITA
            elif emg_leg_choice == "Left":
                video_url = "http://googleusercontent.com/youtube.com/combined_left_leg_video" # SEU VÍDEO COMBINED PERNA ESQUERDA
                diagram_image = "img/combined_left_leg_diagram.png" # SUA IMAGEM COMBINED PERNA ESQUERDA
            else: # Default ou se não houver perna selecionada
                video_url = "http://googleusercontent.com/youtube.com/combined_default_video" 
                diagram_image = "img/combined_default_diagram.png" 

            show_start_collection_button = True
            
        # Define a new_exam_start_message aqui após selected_session_type e emg_leg_choice estarem populados
        if selected_session_type in ["EMG_ONLY", "COMBINED"]:
            new_exam_start_message = (
                f"Exam type chosen: {selected_session_type.replace('_', ' ')} for {emg_leg_choice} leg. "
                f"Duration: {collection_duration_minutes} min (adjustable). Press 'Start Exam' to proceed."
            )
        else: # GAIT_ONLY
            new_exam_start_message = (
                f"Exam type chosen: {selected_session_type.replace('_', ' ')}. "
                f"Duration: {collection_duration_minutes} min (adjustable). Press 'Start Exam' to proceed."
            )
        session_status_message = new_exam_start_message

    context = {
        "patient": patient,
        "message": message,
        "end_message": session_status_message, 
        "notification": notification,
        "notification_pop": notification_pop,
        "is_session_active": is_any_session_active_in_consumer,
        "current_session_db_id": last_session_db_id,
        "collection_duration_minutes": collection_duration_minutes,
        
        "selected_session_type": selected_session_type, # Passa para renderização condicional no HTML
        "emg_leg_choice": emg_leg_choice, # Passa para pré-seleção do dropdown
        "session_type_display_name": session_type_display_name,
        "video_url": video_url,
        "explanation_text": explanation_text,
        "diagram_image": diagram_image,
        "preparation_instructions": preparation_instructions, 
        "show_start_collection_button": show_start_collection_button,
        "show_session_actions_menu": show_session_actions_menu,
    }
    return render(request, "patientsession.html", context)

#-------------------------------------------------------------------------------------------------------------------------------

def delete_session(request, patient_pk, session_id):
    """
    View para eliminar uma sessão específica de um paciente.
    Requer uma requisição POST para segurança.
    """
    if request.method == 'POST':
        try:
            # 1. Encontra o objeto Patient usando a chave primária (pk) passada na URL.
            #    Se o paciente não for encontrado, retorna um erro 404.
            patient = get_object_or_404(Patient, pk=patient_pk)

            # 2. Encontra o objeto Session usando seu ID e garantindo que ele pertence
            #    ao paciente encontrado. Isso é uma importante medida de segurança.
            #    Se a sessão não for encontrada ou não pertencer a este paciente, retorna 404.
            session = get_object_or_404(Sessions, id=session_id, Patient_id=patient_pk)

            # 3. Elimina a sessão do banco de dados.
            session.delete()

            # 4. Adiciona uma mensagem de sucesso que será exibida ao usuário (se o sistema de mensagens estiver configurado).
            messages.success(request, f'Session for {patient.first_name} deleted successfully.')

            # 5. Redireciona o usuário de volta para a página de detalhes do paciente.
            #    'pinfo' é o nome da URL para os detalhes do paciente, e 'id=patient.id'
            #    passa o ID do paciente necessário para essa URL.
            return redirect('pinfo', id=patient.id)

        except Exception as e:
            # Captura qualquer erro que possa ocorrer durante o processo de eliminação.
            messages.error(request, f'Error deleting session: {e}')
            
        
            # Redireciona de volta para a página do paciente, mesmo em caso de erro.
            return redirect('pinfo', id=patient_pk)
    else:
        # Se a requisição não for POST (ex: alguém tentar acessar a URL diretamente no navegador via GET),
        # informa que o método não é permitido e redireciona.
        messages.error(request, 'Invalid request method. Session can only be deleted via POST.')
        return redirect('pinfo', id=patient_pk)

#-------------------------------------------------------------------------------------------------------------------------------
DEBUG_MODE_WITHOUT_ARDUINO = False
#-------------------------------------------------------------------------------------------------------------------------------

def SessionReview(request, id, session_id):
    session = get_object_or_404(Sessions, Patient=id, id=session_id)
    patient = get_object_or_404(Patient, id=id)

    notification = None
    session_going_status = "Waiting for Arduino data..." # Initial collection status

    # Logic to determine session status
    if DEBUG_MODE_WITHOUT_ARDUINO:
        # In debug mode, active/paused status is derived from DB status
        # and the presence of session_start_time.
        is_session_active_in_consumer = (
            session.session_start_time is not None and
            session.session_status_r not in ["Completed", "Paused"] # Assuming "Ongoing" or similar for active
        )
        is_session_paused = (
            session.session_status_r == "Paused" or
            session.session_status_l == "Paused" or
            session.session_status_emg == "Paused"
        )
        # If paused in DB, debug mode should consider it paused
        if is_session_paused:
            is_session_active_in_consumer = False # Not active if paused

        is_data_saved_ack = (session.session_status_r == "Completed") # Simulates ACK if completed
        logger.info(f"DEBUG_MODE: is_session_active_in_consumer={is_session_active_in_consumer}, is_session_paused={is_session_paused}")

    else:
        # Normal behavior, dependent on MQTT consumer
        is_session_active_in_consumer = consumers.get_session_status_from_consumer(session.id)
        is_data_saved_ack = consumers.get_data_saved_ack_status(session.id)

        # Determine if session is paused based on DB status AND if consumer is NOT active
        is_session_paused_db = (
            session.session_status_r == "Paused" or
            session.session_status_l == "Paused" or
            session.session_status_emg == "Paused"
        )
        is_session_paused = is_session_paused_db and not is_session_active_in_consumer


    # 4. Define the status message for display on the page
    if is_session_active_in_consumer and not is_session_paused:
        session_going_status = "Session in progress. (Simulated) Waiting for data." if DEBUG_MODE_WITHOUT_ARDUINO else "Session in progress. Waiting for Arduino data..."
    elif is_session_paused:
        session_going_status = "Session Paused. Data collection suspended."
    elif is_data_saved_ack:
        session_going_status = "Session Completed (data saved and confirmed)."
    elif session.session_status_r == "Completed": # If ACK is not defined, but DB says 'Completed'
        session_going_status = "Session Completed (data saved)."
    else:
        session_going_status = "Session Inactive or Waiting for Data."

    # Calculate total session time only if there is a start time
    # and the session is marked as "Completed" in the DB and does not yet have a recorded time.
    if not session.session_time and session.session_start_time and \
       (session.session_status_r == "Completed" or session.session_status_l == "Completed" or session.session_status_emg == "Completed"):
        end_time_for_calc = session.session_end_time if session.session_end_time else timezone.now()
        session.session_time = (end_time_for_calc - session.session_start_time).total_seconds()
        session.save() # Saves session time to DB

    # --- Data Processing (GAIT and EMG) ---
    # Real-time or DB data loading logic remains the same.
    # In debug mode, since there's no consumer data, it will come from the DB if it exists.

    data_list_gait_r = []
    data_list_gait_l = []
    plot_gait_r = None
    plot_gait_l = None
    plot_expected_gait_r = None
    plot_expected_gait_l = None
    prediction_gait = None

    if session.session_type in ["GAIT_ONLY", "COMBINED"]:
        # Tries to load real-time data (from consumer) if the session is active (or paused, for viewing what has already been collected)
        # OR if in DEBUG_MODE and simulating an active/paused session
        if (is_session_active_in_consumer or is_session_paused) and not DEBUG_MODE_WITHOUT_ARDUINO and \
           hasattr(consumers, 'temp_session_data') and session.id in consumers.temp_session_data:
            current_gait_data = consumers.temp_session_data[session.id].get('gait', {})
            data_list_gait_r = current_gait_data.get('gait_r', [])
            data_list_gait_l = current_gait_data.get('gait_l', [])
            logger.info(f"Session {session.id}: Loading real-time GAIT data from consumer. R:{len(data_list_gait_r)}, L:{len(data_list_gait_l)}")
        else: # If not active/paused or no temporary data, try to load from DB
            if session.session_results_gait_right and session.session_results_gait_left:
                try:
                    # Converts from CSV string back to list of floats
                    data_list_gait_r = [float(x) for x in session.session_results_gait_right.split(',') if x.strip()]
                    data_list_gait_l = [float(x) for x in session.session_results_gait_left.split(',') if x.strip()]
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing GAIT data from DB for session {session.id}: {e}")
                    data_list_gait_r = []
                    data_list_gait_l = []
                logger.info(f"Session {session.id}: Loading GAIT data from DB. R:{len(data_list_gait_r)}, L:{len(data_list_gait_l)}")


        if data_list_gait_r and data_list_gait_l:
            plot_gait_r = generate_plot(x1=list(np.arange(0,len(data_list_gait_r))/100), y_r=data_list_gait_r, side=0)
            plot_gait_l = generate_plot(x1=list(np.arange(0,len(data_list_gait_l))/100), y_l=data_list_gait_l, side=1)

            if 'model' in globals() and model: # Checks if knee model was loaded
                gender = 1 if patient.gender == 'Male' else 0
                with torch.no_grad():
                    knee_angle_curve = model(torch.tensor([[patient.age, gender, patient.weight, patient.height]]))
                smoothed_curve = savgol_filter(knee_angle_curve[0].cpu().detach().numpy(), 15, 4)

                plot_expected_gait_r = generate_expected_curve(x = list(np.arange(0,len(data_list_gait_r))/100), expected_curve = smoothed_curve, y = data_list_gait_r, side = 0)
                plot_expected_gait_l = generate_expected_curve(x = list(np.arange(0,len(data_list_gait_l))/100), expected_curve = smoothed_curve, y = data_list_gait_l, side = 1)
            else:
                logger.warning("Knee model not loaded, skipping expected curve generation.")

        logger.info(f"Right GAIT data processed: {len(data_list_gait_r)} points")
        logger.info(f"Left GAIT data processed: {len(data_list_gait_l)} points")
    # --- Realiza predições com os modelos de machine learning ---
        if model_gait is not None and scaler_gait is not None and data_list_gait_r and data_list_gait_l:
            # Exemplo: Calcular features necessárias para o modelo KOA
            import numpy as np
            from statistics import mean, stdev

            amplitude_r = max(data_list_gait_r) - min(data_list_gait_r)
            amplitude_l = max(data_list_gait_l) - min(data_list_gait_l)
            amplitude_avg = (amplitude_r + amplitude_l) / 2

            features_gait = {
                "Age": patient.age,
                "Height_cm": patient.height,
                "Mean_Knee_Max": mean([max(data_list_gait_r), max(data_list_gait_l)]),
                "Mean_Knee_Min": mean([min(data_list_gait_r), min(data_list_gait_l)]),
                "Amplitude": amplitude_avg,
                "Std_Max": stdev([max(data_list_gait_r), max(data_list_gait_l)]),
                "Std_Min": stdev([min(data_list_gait_r), min(data_list_gait_l)])
            }

            df_input = pd.DataFrame([features_gait])
            X_scaled = scaler_gait.transform(df_input)

            prediction_gait = model_gait.predict(X_scaled)[0]
            logger.info(f"🔍 Predição RF KOA: {prediction_gait}")  # 0 = NM, 1 = KOA

        if model_severity is not None and prediction_gait == 1:
            dmatrix = xgb.DMatrix(X_scaled)
            severity_pred = model_severity.predict(dmatrix)
            severity_class = int(np.argmax(severity_pred))  # ou np.round(severity_pred) se for regressão

            logger.info(f"🔍 Predição Severidade KOA: {severity_class}")
            context["koa_prediction"] = "KOA" if prediction_gait else "Normal"
            context["severity_prediction"] = severity_class if prediction_gait else None

    # --- EMG Data Processing (3 Channels) ---
    data_list_emg_channel1 = []
    data_list_emg_channel2 = []
    data_list_emg_channel3 = []
    emg_prediction_text = session.emg_prediction if session.emg_prediction else "N/A" # Uses saved prediction first

    if session.session_type in ["EMG_ONLY", "COMBINED"]:
        if (is_session_active_in_consumer or is_session_paused) and not DEBUG_MODE_WITHOUT_ARDUINO and \
           hasattr(consumers, 'temp_session_data') and session.id in consumers.temp_session_data:
            current_emg_data = consumers.temp_session_data[session.id].get('emg', {})
            data_list_emg_channel1 = current_emg_data.get('emg1', [])
            data_list_emg_channel2 = current_emg_data.get('emg2', [])
            data_list_emg_channel3 = current_emg_data.get('emg3', [])
            logger.info(f"Session {session.id}: Loading real-time EMG data from consumer. C1:{len(data_list_emg_channel1)}, C2:{len(data_list_emg_channel2)}, C3:{len(data_list_emg_channel3)}")
        else:
            if session.session_results_emg_channel1:
                try:
                    data_list_emg_channel1 = json.loads(session.session_results_emg_channel1)
                    data_list_emg_channel2 = json.loads(session.session_results_emg_channel2)
                    data_list_emg_channel3 = json.loads(session.session_results_emg_channel3)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing EMG data from DB for session {session.id}: {e}")
                    data_list_emg_channel1 = []
                    data_list_emg_channel2 = []
                    data_list_emg_channel3 = []
        logger.info(f"Session {session.id}: Loading EMG data from DB. C1:{len(data_list_emg_channel1)}, C2:{len(data_list_emg_channel2)}, C3:{len(data_list_emg_channel3)}")

        # --- Lógica de Predição EMG ---
        if all([data_list_emg_channel1, data_list_emg_channel2, data_list_emg_channel3]):
            min_len_emg = min(len(data_list_emg_channel1), len(data_list_emg_channel2), len(data_list_emg_channel3))
            
                #np.array(data_list_emg_channel1)[:min_len_emg],
                #np.array(data_list_emg_channel2)[:min_len_emg],
                #np.array(data_list_emg_channel3)[:min_len_emg]
                # Pré-processa os canais individualmente
            filtered_emg1 = preprocess_emg_signal(data_list_emg_channel1[:min_len_emg], fs=1500)
            filtered_emg2 = preprocess_emg_signal(data_list_emg_channel2[:min_len_emg], fs=1500)
            filtered_emg3 = preprocess_emg_signal(data_list_emg_channel3[:min_len_emg], fs=1500)

            # Junta os canais já pré-processados
            raw_emg_data_for_preprocessing = np.column_stack((
                filtered_emg1,
                filtered_emg2,
                filtered_emg3
            ))
            

            side_numeric = 0 # Valor numérico para Esquerda (se o modelo foi treinado com 0 para Esquerda)
            if session.emg_measured_leg == 'Right': # Corrigido para 'Right' ou 'Left'
                side_numeric = 1 # Valor numérico para Direita (se o modelo foi treinado com 1 para Direita)

            # Crie o vetor one-hot [1,0] (para Esquerda) ou [0,1] (para Direita)
            side_onehot_feature = tf.keras.utils.to_categorical(side_numeric, num_classes=2) # Shape (2,)

            # Repita este vetor para cada timestep na janela de dados
            side_expanded = np.repeat(side_onehot_feature[np.newaxis, :], min_len_emg, axis=0)

            # Concatene os dados EMG e a feature de Lado
            X_augmented_input = np.concatenate([raw_emg_data_for_preprocessing, side_expanded], axis=1)

            emg_model_to_use = None
            emg_mean_to_use = None
            emg_std_to_use = None

            emg_model_type_to_use = None

            if patient.dominant_leg_side and session.emg_measured_leg:
                # Corrigido para comparar 'Right'/'Left'
                if patient.dominant_leg_side == session.emg_measured_leg: # Corrigido para comparar corretamente
                    emg_model_type_to_use = 'D' # A perna medida é a dominante do paciente
                else:
                    emg_model_type_to_use = 'ND' # A perna medida é a não-dominante do paciente
                logger.info(f"Perna medida na sessão ({session.emg_measured_leg}) é classificada como {emg_model_type_to_use} (perna dominante do paciente é {patient.dominant_leg_side}).")
            else:
                emg_model_type_to_use = 'D' # PADRÃO: Assume dominante se a informação estiver faltando
                logger.warning(f"Lado da perna dominante do paciente ({patient.dominant_leg_side}) ou perna medida na sessão ({session.emg_measured_leg}) não definidos. Usando modelo Dominante como fallback.")

            # Certifique-se de que as variáveis loaded_emg_*_model/mean/std estão disponíveis no escopo global ou via import
            # Ex: (no topo do seu views.py)
            # from django.conf import settings
            # try:
            #     loaded_emg_dom_model = joblib.load(settings.EMG_DOM_MODEL_PATH)
            #     loaded_emg_dom_mean = joblib.load(settings.EMG_DOM_MEAN_PATH)
            #     loaded_emg_dom_std = joblib.load(settings.EMG_DOM_STD_PATH)
            #     loaded_emg_ndom_model = joblib.load(settings.EMG_NDOM_MODEL_PATH)
            #     loaded_emg_ndom_mean = joblib.load(settings.EMG_NDOM_MEAN_PATH)
            #     loaded_emg_ndom_std = joblib.load(settings.EMG_NDOM_STD_PATH)
            # except Exception as e:
            #     logger.error(f"Erro ao carregar modelos EMG: {e}")
            #     loaded_emg_dom_model = loaded_emg_dom_mean = loaded_emg_dom_std = None
            #     loaded_emg_ndom_model = loaded_emg_ndom_mean = loaded_emg_ndom_std = None


            if emg_model_type_to_use == 'D' and 'loaded_emg_dom_model' in globals() and loaded_emg_dom_model:
                emg_model_to_use = loaded_emg_dom_model
                emg_mean_to_use = loaded_emg_dom_mean
                emg_std_to_use = loaded_emg_dom_std
                logger.info("Usando modelo EMG Dominante.")
            elif emg_model_type_to_use == 'ND' and 'loaded_emg_ndom_model' in globals() and loaded_emg_ndom_model:
                emg_model_to_use = loaded_emg_ndom_model
                emg_mean_to_use = loaded_emg_ndom_mean
                emg_std_to_use = loaded_emg_ndom_std
                logger.info("Usando modelo EMG Não-Dominante.")
            else:
                if 'loaded_emg_dom_model' in globals() and loaded_emg_dom_model: # Tenta usar o dominante como último recurso
                    emg_model_to_use = loaded_emg_dom_model
                    emg_mean_to_use = loaded_emg_dom_mean
                    emg_std_to_use = loaded_emg_dom_std
                    logger.warning("Modelo específico não carregado ou dominância não determinada. Usando modelo Dominante como fallback.")
                else:
                    logger.error("Nenhum modelo EMG disponível para inferência.")
                    emg_prediction_text = "Erro: Modelo EMG não disponível."

            if emg_model_to_use and emg_mean_to_use is not None and emg_std_to_use is not None:
                fixed_timesteps = 384 # AJUSTE ESTE VALOR PARA O SEU MODELO!

                if min_len_emg < fixed_timesteps:
                    logger.warning(f"Dados EMG insuficientes ({min_len_emg} pontos) para predição. Necessário {fixed_timesteps}.")
                    emg_prediction_text = "Dados insuficientes para predição EMG."
                else:
                    # Pré-processamento: normalização
                    processed_emg_data_norm = (X_augmented_input - emg_mean_to_use) / emg_std_to_use

                    # Pegar os últimos `fixed_timesteps` pontos para o input do modelo
                    input_for_model = processed_emg_data_norm[-fixed_timesteps:, :][np.newaxis, :, :] # Adiciona dimensão de batch

                    try:
                        emg_raw_prediction = emg_model_to_use.predict(input_for_model)

                        emg_predicted_class = (emg_raw_prediction > 0.5).astype("int32")[0][0] # Assume saída binária
                        emg_prediction_text = "Fadiga Muscular" if emg_predicted_class == 1 else "Normal"
                        session.emg_prediction = emg_prediction_text # Atualiza no DB se foi calculado agora
                        session.save() # Salva a predição no DB
                        logger.info(f"Predição EMG: {emg_prediction_text} (Probabilidade: {emg_raw_prediction[0][0]:.4f})")
                    except Exception as e:
                        logger.error(f"Erro durante a predição EMG: {e}")
                        emg_prediction_text = f"Erro na predição: {e}"
            else:
                # Se não há modelo ou escaladores, ou se a predição já foi feita antes, use o valor salvo
                emg_prediction_text = session.emg_prediction if session.emg_prediction else "Sem predição (modelo não carregado)."
        else:
            # Se não há dados EMG ou eles estão incompletos
            emg_prediction_text = session.emg_prediction if session.emg_prediction else "Sem predição (dados EMG ausentes)."


    # Session action buttons logic (End, Progress, Pause, Resume)
    if request.method == "POST":
        channel_layer = get_channel_layer()

        if 'progress' in request.POST:
            session_notes = request.POST.get('session_notes', '')
            session.session_notes = session_notes
            session.save()
            notification = "Session notes saved."

            if is_session_active_in_consumer:
                notification = "Data collection in progress. Charts will be updated with the latest data."
            else:
                if is_data_saved_ack:
                    notification = f"Session {session.id} data has been saved to the database. You can review the data."
                    session_going_status = "Session Completed (data saved)."
                else:
                    notification = "Session is not active and data is not yet confirmed as saved. Please try again in a few seconds."
                    session_going_status = "Waiting for saving or data..."


        elif 'end_session' in request.POST:
            session_notes = request.POST.get('session_notes', '')
            session.session_notes = session_notes
            session.session_status = 'COMPLETED'  # ou o status que indicar que não está mais ativa
            session.is_session_active = False 
            session.save()
            notification = "Session notes saved."

            # In DEBUG_MODE, we simulate the end of the session only in the DB
            if DEBUG_MODE_WITHOUT_ARDUINO or is_session_active_in_consumer or is_session_paused:
                if not DEBUG_MODE_WITHOUT_ARDUINO: # If not in debug mode, send MQTT
                    async_to_sync(channel_layer.send)(
                        "mqtt.publish",
                        {
                            "type": "mqtt_publish",
                            "action": "publish",
                            "topic": "django/confirm",
                            "payload": "End",
                            "qos": 2,
                        }
                    )
                    logger.info(f"Command 'End' sent to Arduino for session {session.id}.")

                session.session_end_time = timezone.now()
                if session.session_start_time:
                    session.session_time = (session.session_end_time - session.session_start_time).total_seconds()

                session.session_status_r = "Completed"
                session.session_status_l = "Completed"
                session.session_status_emg = "Completed"

                # For DEBUG_MODE, we can simulate that data has been "saved"
                if DEBUG_MODE_WITHOUT_ARDUINO:
                    # Optional: for DEBUG_MODE, you can simulate some data for the plot
                    # session.session_results_r = json.dumps([i for i in range(100)])
                    # session.session_results_l = json.dumps([i*0.5 for i in range(100)])
                    pass # No temporary data to save without the real consumer

                elif hasattr(consumers, 'temp_session_data') and session.id in consumers.temp_session_data:
                    temp_gait_data = consumers.temp_session_data[session.id].get('gait', {})
                    temp_emg_data = consumers.temp_session_data[session.id].get('emg', {})

                    if 'gait_r' in temp_gait_data:
                        session.session_results_gait_right = ','.join(map(str, temp_gait_data['gait_r']))
                    if 'gait_l' in temp_gait_data:
                        session.session_results_gait_left = ','.join(map(str, temp_gait_data['gait_l']))
                    if 'emg1' in temp_emg_data:
                        session.session_results_emg_channel1 = json.dumps(temp_emg_data['emg1'])
                    if 'emg2' in temp_emg_data:
                        session.session_results_emg_channel2 = json.dumps(temp_emg_data['emg2'])
                    if 'emg3' in temp_emg_data:
                        session.session_results_emg_channel3 = json.dumps(temp_emg_data['emg3'])

                    consumers.clear_session_data(session.id)
                    logger.info(f"Temporary data cleared for session {session.id}.")

                session.save()

                if not DEBUG_MODE_WITHOUT_ARDUINO:
                    async_to_sync(set_active_status_for_session)(session.id, False)
                    consumers.set_data_saved_ack_status(session.id, True)

                notification = "Session completed and data saved successfully!"
                session_going_status = "Session Completed (data saved)."
                logger.info(f"Session {session.id} ended and data saved. Redirecting to Patient Information.")
                return redirect(reverse('sessionreview', kwargs={'id': patient.id, 'session_id': session.id}))
            else:
                notification = "No active or paused session to end."
                logger.warning(f"Attempt to end inactive or already ended session {session.id}.")


        elif 'pause_session' in request.POST:
            # In DEBUG_MODE, we simulate pausing the session only in the DB
            session_notes = request.POST.get('session_notes', '')
            session.session_notes = session_notes             
            session.save()
            notification = "Session notes saved."

            if (DEBUG_MODE_WITHOUT_ARDUINO and session.session_status_r != "Paused" and session.session_status_r != "Completed") or \
               (is_session_active_in_consumer and not is_session_paused):

                if not DEBUG_MODE_WITHOUT_ARDUINO:
                    async_to_sync(channel_layer.send)(
                        "mqtt.publish",
                        {
                            "type": "mqtt_publish",
                            "action": "publish",
                            "topic": "django/confirm",
                            "payload": "Pause",
                            "qos": 2,
                        }
                    )
                    logger.info(f"Command 'Pause' sent to Arduino for session {session.id}.")
                    async_to_sync(set_active_status_for_session)(session.id, False)

                session.session_status_r = "Paused"
                session.session_status_l = "Paused"
                session.session_status_emg = "Paused"
                session.save()

                notification = "Session paused. Data collection suspended."
                session_going_status = "Session Paused."
                 # Updates the local variable for the current request context
                return redirect('sessionreview', id=patient.id, session_id=session.id)
            else:
                notification = "No active session to pause, or already paused."
                logger.warning(f"Attempt to pause inactive or already paused session {session.id}.")

        elif 'resume_session' in request.POST:
            session_notes = request.POST.get('session_notes', '')
            session.session_notes = session_notes             
            session.save()
            # In DEBUG_MODE, we simulate resuming the session only in the DB
            if (DEBUG_MODE_WITHOUT_ARDUINO and session.session_status_r == "Paused") or is_session_paused:

                if not DEBUG_MODE_WITHOUT_ARDUINO:
                    async_to_sync(channel_layer.send)(
                        "mqtt.publish",
                        {
                            "type": "mqtt_publish",
                            "action": "publish",
                            "topic": "django/confirm",
                            "payload": "Resume",
                            "qos": 2,
                        }
                    )
                    logger.info(f"Command 'Resume' sent to Arduino for session {session.id}.")
                    async_to_sync(set_active_status_for_session)(session.id, True)

                session.session_status_r = "Ongoing"
                session.session_status_l = "Ongoing"
                session.session_status_emg = "Ongoing"
                session.save()

                notification = "Session resumed. Data collection restarted."
                session_going_status = "Session in progress. Waiting for Arduino data..."
                 # Updates the local variable for the current request context
                return redirect('sessionreview', id=patient.id, session_id=session.id)
            else:
                notification = "No paused session to resume."
                logger.warning(f"Attempt to resume a session that is not in a paused state {session.id}.")

        elif 'restart_session' in request.POST:
            if not DEBUG_MODE_WITHOUT_ARDUINO: # Only send MQTT if not in debug mode
                if is_session_active_in_consumer or is_session_paused:
                    async_to_sync(channel_layer.send)(
                        "mqtt.publish",
                        {
                            "type": "mqtt_publish",
                            "action": "publish",
                            "topic": "django/confirm",
                            "payload": "End",
                            "qos": 2,
                        }
                    )
                    logger.info(f"Command 'End' sent to restart session {session.id}.")
                    consumers.clear_session_data(session.id)
                    async_to_sync(set_active_status_for_session)(session.id, False)
                    consumers.set_data_saved_ack_status(session.id, False)
                notification = "Current session ended. Redirecting to start a new session..."
            else:
                # In DEBUG_MODE, just redirect and logically clear the DB to restart
                session.session_status_r = "Awaiting" # Or other initial status
                session.session_status_l = "Awaiting"
                session.session_status_emg = "Awaiting"
                session.session_start_time = None
                session.session_end_time = None
                session.session_time = None
                session.save()
                notification = "Current (simulated) session ended. Redirecting to start a new session."

            return redirect('new_session', id=patient.id)

       
        elif 'save_edited_gai_data' in request.POST:
            edited_gai_data_r = request.POST.get('gai_data_r', '')
            edited_gai_data_l = request.POST.get('gai_data_l', '')
            # Validação e processamento dos dados
            if edited_gai_data_r:
                session.session_results_gait_right = edited_gai_data_r
            if edited_gai_data_l:
                session.session_results_gait_left = edited_gai_data_l
            session.save()
            notification = "Dados de GAI editados salvos com sucesso!"
            logger.info(f"Dados de GAI editados salvos para a sessão {session.id}.")
            return JsonResponse({'status': 'success', 'message': notification})

        elif 'save_edited_emg_data' in request.POST:
            edited_emg_channel1 = request.POST.get('emg_channel1', '')
            edited_emg_channel2 = request.POST.get('emg_channel2', '')
            edited_emg_channel3 = request.POST.get('emg_channel3', '')
            # Validação e processamento dos dados
            if edited_emg_channel1:
                session.session_results_emg_channel1 = edited_emg_channel1
            if edited_emg_channel2:
                session.session_results_emg_channel2 = edited_emg_channel2
            if edited_emg_channel3:
                session.session_results_emg_channel3 = edited_emg_channel3
            session.save()
            notification = "Dados de EMG editados salvos com sucesso!"
            logger.info(f"Dados de EMG editados salvos para a sessão {session.id}.")
            return JsonResponse({'status': 'success', 'message': notification})
        elif 'go_back' in request.POST:
            # In DEBUG_MODE, we can be more flexible to go back
            if DEBUG_MODE_WITHOUT_ARDUINO or (not is_session_active_in_consumer and not is_session_paused):
                return redirect('pinfo', id=patient.id)
            else:
                notification = "Please wait for the session to end or terminate it to go back."
        elif 'save_notes' in request.POST:
            session_notes = request.POST.get('session_notes', '')
            session.session_notes = session_notes             
            session.save() # Apenas salva, pois as notas já foram atualizadas acima
            notification = "Session notes saved."

    # Context to render the template
    context = {
        "patient": patient,
        "session": session,
        "notification": notification,
        "session_going_status": session_going_status,
        "is_session_active": is_session_active_in_consumer,
        "is_session_paused": is_session_paused,
        "plot_gait_r": plot_gait_r,
        "plot_gait_l": plot_gait_l,
        "plot_expected_gait_r": plot_expected_gait_r,
        "plot_expected_gait_l": plot_expected_gait_l,
        "data_list_emg_channel1": json.dumps(data_list_emg_channel1), # JSON-encode para uso no JS no template
        "data_list_emg_channel2": json.dumps(data_list_emg_channel2), # JSON-encode para uso no JS no template
        "data_list_emg_channel3": json.dumps(data_list_emg_channel3), # JSON-encode para uso no JS no template
        "emg_prediction_text": emg_prediction_text,
    }
    return render(request, "sessionreview.html", context)

# --- Funções para obter dados via AJAX (para gráficos dinâmicos no frontend) ---

# GAI functions
def get_data_points(request, start, count, id, session_id):
    session = get_object_or_404(Sessions, Patient=id, id=session_id)
    # Tenta obter dados em tempo real do consumer se a sessão estiver ativa
    is_session_active = False
    if hasattr(consumers, 'active_sessions_status') and session_id in consumers.active_sessions_status:
        is_session_active = consumers.active_sessions_status[session_id]

    if is_session_active and hasattr(consumers, 'active_session_data') and session_id in consumers.active_session_data:
        current_gait_data = consumers.active_session_data[session_id].get('gait', {})
        data_list_r = current_gait_data.get('right', [])
        data_list_l = current_gait_data.get('left', [])
        logger.debug(f"AJAX: Servindo dados GAI em tempo real para sessão {session_id}. R:{len(data_list_r)}, L:{len(data_list_l)}")
    else:
        data_r_str = session.session_results_gait_right.strip('[]') if session.session_results_gait_right else ""
        data_l_str = session.session_results_gait_left.strip('[]') if session.session_results_gait_left else ""
        data_list_r = [float(i) for i in data_r_str.split(',') if i.strip()]
        data_list_l = [float(i) for i in data_l_str.split(',') if i.strip()]
        logger.debug(f"AJAX: Servindo dados GAI do DB para sessão {session_id}. R:{len(data_list_r)}, L:{len(data_list_l)}")
    
    end = start + count
    data_points_r = [{'timestamp': i, 'value': data_list_r[i]} for i in range(start, min(end, len(data_list_r)))]
    data_points_l = [{'timestamp': i, 'value': data_list_l[i]} for i in range(start, min(end, len(data_list_l)))]

    return JsonResponse({'right': data_points_r, 'left': data_points_l})

def get_expected_curve(request, id):
    patient = get_object_or_404(Patient, id=id)
    smoothed_curve = []
    if model:
        sex = 1 if patient.sex == 'Male' else 0
        with torch.no_grad():
            knee_angle_curve = model(torch.tensor([[patient.age, sex, patient.weight, patient.height]]))
        smoothed_curve = savgol_filter(knee_angle_curve[0].cpu().detach().numpy(), 15, 4)
    return JsonResponse({'expected_curve': smoothed_curve.tolist()})


def get_data_between_indexes_right(request, id, session_id, start_index, end_index):
    session = get_object_or_404(Sessions, Patient=id, id=session_id)
    # Tenta obter dados em tempo real do consumer se a sessão estiver ativa
    is_session_active = False
    if hasattr(consumers, 'active_sessions_status') and session_id in consumers.active_sessions_status:
        is_session_active = consumers.active_sessions_status[session_id]

    if is_session_active and hasattr(consumers, 'active_session_data') and session_id in consumers.active_session_data:
        current_gait_data = consumers.active_session_data[session_id].get('gait', {})
        data_list_r = current_gait_data.get('right', [])
        logger.debug(f"AJAX: Servindo slice GAI Direita em tempo real para sessão {session_id}. Len:{len(data_list_r)}")
    else:
        data_r_str = session.session_results_gait_right.strip('[]') if session.session_results_gait_right else ""
        data_list_r = [float(i) for i in data_r_str.split(',') if i.strip()]
        logger.debug(f"AJAX: Servindo slice GAI Direita do DB para sessão {session_id}. Len:{len(data_list_r)}")

    right_leg_data = data_list_r[start_index:end_index]
    return JsonResponse({'right': right_leg_data})

def get_data_between_indexes_left(request, id, session_id, start_index, end_index):
    session = get_object_or_404(Sessions, Patient=id, id=session_id)
    # Tenta obter dados em tempo real do consumer se a sessão estiver ativa
    is_session_active = False
    if hasattr(consumers, 'active_sessions_status') and session_id in consumers.active_sessions_status:
        is_session_active = consumers.active_sessions_status[session_id]

    if is_session_active and hasattr(consumers, 'active_session_data') and session_id in consumers.active_session_data:
        current_gait_data = consumers.active_session_data[session_id].get('gait', {})
        data_list_l = current_gait_data.get('left', [])
        logger.debug(f"AJAX: Servindo slice GAI Esquerda em tempo real para sessão {session_id}. Len:{len(data_list_l)}")
    else:
        data_l_str = session.session_results_gait_left.strip('[]') if session.session_results_gait_left else ""
        data_list_l = [float(i) for i in data_l_str.split(',') if i.strip()]
        logger.debug(f"AJAX: Servindo slice GAI Esquerda do DB para sessão {session_id}. Len:{len(data_list_l)}")

    left_leg_data = data_list_l[start_index:end_index]
    return JsonResponse({'left': left_leg_data})

def get_all_data_points(request, id, session_id):
    session = get_object_or_404(Sessions, Patient=id, id=session_id)
    # Tenta obter dados em tempo real do consumer se a sessão estiver ativa
    is_session_active = False
    if hasattr(consumers, 'active_sessions_status') and session_id in consumers.active_sessions_status:
        is_session_active = consumers.active_sessions_status[session_id]

    if is_session_active and hasattr(consumers, 'active_session_data') and session_id in consumers.active_session_data:
        current_gait_data = consumers.active_session_data[session_id].get('gait', {})
        data_list_r = current_gait_data.get('right', [])
        data_list_l = current_gait_data.get('left', [])
        logger.debug(f"AJAX: Servindo TODOS os dados GAI em tempo real para sessão {session_id}. R:{len(data_list_r)}, L:{len(data_list_l)}")
    else:
        data_r_str = session.session_results_gait_right.strip('[]') if session.session_results_gait_right else ""
        data_l_str = session.session_results_gait_left.strip('[]') if session.session_results_gait_left else ""
        data_list_r = [float(i) for i in data_r_str.split(',') if i.strip()]
        data_list_l = [float(i) for i in data_l_str.split(',') if i.strip()]
        logger.debug(f"AJAX: Servindo TODOS os dados GAI do DB para sessão {session_id}. R:{len(data_list_r)}, L:{len(data_list_l)}")

    return JsonResponse({'right': data_list_r, 'left': data_list_l})

# --- NOVAS FUNÇÕES PARA OBTER DADOS EMG VIA AJAX ---

def get_emg_data_channel(request, id, session_id, channel_num):
    session = get_object_or_404(Sessions, Patient=id, id=session_id)
    data_list_emg = []
    
    # Tenta obter dados em tempo real do consumer se a sessão estiver ativa
    is_session_active = False
    if hasattr(consumers, 'active_sessions_status') and session_id in consumers.active_sessions_status:
        is_session_active = consumers.active_sessions_status[session_id]

    if is_session_active and hasattr(consumers, 'active_session_data') and session_id in consumers.active_session_data:
        current_emg_data = consumers.active_session_data[session_id].get('emg', {})
        if channel_num == 1:
            data_list_emg = current_emg_data.get('channel1', [])
        elif channel_num == 2:
            data_list_emg = current_emg_data.get('channel2', [])
        elif channel_num == 3:
            data_list_emg = current_emg_data.get('channel3', [])
        logger.debug(f"AJAX: Servindo dados EMG Canal {channel_num} em tempo real para sessão {session_id}. Len:{len(data_list_emg)}")
    else: # Caso contrário, carrega os dados salvos do DB
        if channel_num == 1:
            if session.session_results_emg_channel1:
                try:
                    data_list_emg = json.loads(session.session_results_emg_channel1)
                except json.JSONDecodeError:
                    logger.error(f"Erro ao decodificar JSON para EMG canal 1 na sessão {session_id}")
        elif channel_num == 2:
            if session.session_results_emg_channel2:
                try:
                    data_list_emg = json.loads(session.session_results_emg_channel2)
                except json.JSONDecodeError:
                    logger.error(f"Erro ao decodificar JSON para EMG canal 2 na sessão {session_id}")
        elif channel_num == 3:
            if session.session_results_emg_channel3:
                try:
                    data_list_emg = json.loads(session.session_results_emg_channel3)
                except json.JSONDecodeError:
                    logger.error(f"Erro ao decodificar JSON para EMG canal 3 na sessão {session_id}")
        else:
            return JsonResponse({'error': 'Número de canal inválido.'}, status=400)
        logger.debug(f"AJAX: Servindo dados EMG Canal {channel_num} do DB para sessão {session_id}. Len:{len(data_list_emg)}")
        
    return JsonResponse({f'emg_channel{channel_num}': data_list_emg})


def get_emg_data_between_indexes_channel(request, id, session_id, channel_num, start_index, end_index):
    session = get_object_or_404(Sessions, Patient=id, id=session_id)
    data_list_emg = []
    
    # Tenta obter dados em tempo real do consumer se a sessão estiver ativa
    is_session_active = False
    if hasattr(consumers, 'active_sessions_status') and session_id in consumers.active_sessions_status:
        is_session_active = consumers.active_sessions_status[session_id]

    if is_session_active and hasattr(consumers, 'active_session_data') and session_id in consumers.active_session_data:
        current_emg_data = consumers.active_session_data[session_id].get('emg', {})
        if channel_num == 1:
            data_list_emg = current_emg_data.get('channel1', [])
        elif channel_num == 2:
            data_list_emg = current_emg_data.get('channel2', [])
        elif channel_num == 3:
            data_list_emg = current_emg_data.get('channel3', [])
        logger.debug(f"AJAX: Servindo slice EMG Canal {channel_num} em tempo real para sessão {session_id}. Len:{len(data_list_emg)}")
    else: # Caso contrário, carrega os dados salvos do DB
        if channel_num == 1:
            if session.session_results_emg_channel1:
                try:
                    data_list_emg = json.loads(session.session_results_emg_channel1)
                except json.JSONDecodeError:
                    logger.error(f"Erro ao decodificar JSON para EMG canal 1 na sessão {session_id}")
        elif channel_num == 2:
            if session.session_results_emg_channel2:
                try:
                    data_list_emg = json.loads(session.session_results_emg_channel2)
                except json.JSONDecodeError:
                    logger.error(f"Erro ao decodificar JSON para EMG canal 2 na sessão {session_id}")
        elif channel_num == 3:
            if session.session_results_emg_channel3:
                try:
                    data_list_emg = json.loads(session.session_results_emg_channel3)
                except json.JSONDecodeError:
                    logger.error(f"Erro ao decodificar JSON para EMG canal 3 na sessão {session_id}")
        else:
            return JsonResponse({'error': 'Número de canal inválido.'}, status=400)
        logger.debug(f"AJAX: Servindo slice EMG Canal {channel_num} do DB para sessão {session_id}. Len:{len(data_list_emg)}")
    
    # Retorna o slice dos dados
    return JsonResponse({f'emg_channel{channel_num}': data_list_emg[start_index:end_index]})
