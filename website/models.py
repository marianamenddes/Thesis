#from django.contrib.postgres.fields import ArrayField
from django.db import models
import random
from django.utils import timezone
from django.contrib.auth.hashers import make_password, check_password

from django.contrib.auth.models import AbstractUser
# Create your models here.


class Doctor(AbstractUser): # MUDA AQUI: Herda de AbstractUser
    name = models.CharField(max_length=100, blank=True, null=True)
    id_number = models.CharField(max_length=100, unique=True) # Torna-o obrigatório se for o username
    phone_number = models.IntegerField(null=True, blank=True) # Adicionei blank=True para flexibilidade

    # Se quiseres que 'id_number' seja o campo de login em vez de 'username'
    USERNAME_FIELD = 'id_number'
    # Define campos obrigatórios além do USERNAME_FIELD quando se cria um superuser
    REQUIRED_FIELDS = ['username', 'email'] # 'username' aqui refere-se ao campo herdado de AbstractUser,

    def __str__(self):
        # Decide o que queres que seja a representação string
        if self.name:
            return f"{self.name}"
        elif self.username:
            return self.username
        else:
            return str(self.id_number) # Se id_number for o principal
    

class Patient(models.Model):
    name = models.CharField(max_length=100)
    patient_number = models.IntegerField(null=True)
    phone_number = models.IntegerField(null=True)
    height = models.FloatField(null=True)
    weight = models.FloatField(null=True)
    age = models.IntegerField(null=True)
    sex = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female')])
    doc = models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True)
    session_num = models.IntegerField(null=True)
    dominant_leg_side = models.CharField(max_length=1, choices=[('R', 'Right'), ('L', 'Left')], null=True, blank=True,  help_text="What is the patient's dominant leg side?")
    
    
    def __str__(self):
        return f"{self.name} {self.patient_number}"


class Sessions(models.Model):

    Patient = models.ForeignKey(Patient,on_delete=models.CASCADE)
    session_id = models.IntegerField(null=True)
   
   # Tipo de Sessão: consolidado com choices
    SESSION_TYPES = [
        ('GAIT_ONLY', 'Gait Only'),
        ('EMG_ONLY', 'EMG Only'),
        ('COMBINED', 'Gait and EMG Combined'),
    ]
    session_type = models.CharField(max_length=20, choices=SESSION_TYPES, null=False, default='GAIT_ONLY')

    # Tempos da Sessão: consolidado e clarificado
    date = models.DateTimeField(default=timezone.now) # Data/hora de criação do registo da sessão
    session_start_time = models.DateTimeField(null=True, blank=True) # Início real da recolha de dados (pode ser definido depois)
    session_end_time = models.DateTimeField(null=True, blank=True) # Fim da recolha de dados
    
    collection_duration_minutes = models.IntegerField(default=5) # Duração esperada da coleção
    session_notes = models.TextField(blank=True, null=True)

   # Dados brutos das sessões (GAI - Gait Analysis Instrument)
    session_results_gait_right = models.TextField(blank=True, null=True, default="[]")
    session_results_gait_left = models.TextField(blank=True, null=True, default="[]")
    # Dados brutos das sessões (EMG - Electromyography)
    session_results_emg_channel1 = models.TextField(blank=True, null=True, default="[]")
    session_results_emg_channel2 = models.TextField(blank=True, null=True, default="[]")
    session_results_emg_channel3 = models.TextField(blank=True, null=True, default="[]")

    session_status_l = models.CharField(max_length=20, null=True, default="Ongoing")
    session_status_r = models.CharField(max_length=20, null=True, default="Ongoing")
    session_status_emg = models.CharField(max_length=50, null=False, default="Pending")
    
    
    # Previsão do Modelo EMG
    emg_prediction = models.CharField(max_length=255, null=True, blank=True)
    
    notes = models.TextField(null=True, blank=True)
    session_time = models.IntegerField(null=True, blank=True) # Se este campo for a duração total em tempo (e.g. segundos), o blank=True é útil.

    # Perna Medida (EMG): consolidado com choices e max_length=1
    emg_measured_leg = models.CharField(max_length=1, choices=[('R', 'Right'), ('L', 'Left')], null=True, blank=True)

    # --- Campos de Status ---

    # Status dos Componentes Individuais (Encoders L/R, EMG)
    # Estes campos servem para detalhar o estado de cada fonte de dados.
    # Ex: se o encoder esquerdo está 'Ongoing', 'Disconnected', 'Error'.
    # O default "Ongoing" para L e R faz sentido se eles estiverem sempre prontos para começar.
    # Para EMG, "Pending" pode indicar que ainda não começou ou está a aguardar calibração/configuração.
   
    # Status Geral da Sessão
    # Este campo é o que controla o fluxo principal da sessão (pausar/retomar/finalizar)
    # e a exibição do botão na interface de utilizador.
    # É uma visão de alto nível do estado da sessão.
    SESSION_OVERALL_STATUS_CHOICES = [
        ('Pending', 'Pending Setup'),
        ('Ongoing', 'Ongoing Collection'),
        ('Paused', 'Paused Collection'),
        ('Completed', 'Collection Completed'),
        ('Error', 'Error During Collection'),
    ]
    overall_status = models.CharField(
        max_length=50,
        choices=SESSION_OVERALL_STATUS_CHOICES,
        default='Pending'
    )
    # 1. Para armazenar dados GAI e EMG *editados* pelo utilizador na interface
    session_edited_gait_right = models.TextField(blank=True, null=True, default="[]")
    session_edited_gait_left = models.TextField(blank=True, null=True, default="[]")
    session_edited_emg_channel1 = models.TextField(blank=True, null=True, default="[]")
    session_edited_emg_channel2 = models.TextField(blank=True, null=True, default="[]")
    session_edited_emg_channel3 = models.TextField(blank=True, null=True, default="[]")

    # 2. Para armazenar dados EMG *processados* (e.g., filtrados, retificados)
    # ALTERADO: ArrayField para TextField + default="[]"
    session_processed_emg_channel1 = models.TextField(blank=True, null=True, default="[]")
    session_processed_emg_channel2 = models.TextField(blank=True, null=True, default="[]")
    session_processed_emg_channel3 = models.TextField(blank=True, null=True, default="[]")

    class Meta:
        # Garante que não há sessões duplicadas para o mesmo paciente com o mesmo ID
        unique_together = ('Patient', 'session_id')
        ordering = ['-date'] # Ordena as sessões mais recentes primeiro

    def __str__(self):
        return f"Session {self.session_id} for Patient {self.Patient.name} on {self.date.strftime('%Y-%m-%d')}"
    def get_gait_right_data(self):
        return json.loads(self.session_results_gait_right) if self.session_results_gait_right else []

    def set_gait_right_data(self, data_list):
        self.session_results_gait_right = json.dumps(data_list)

    # Repetir get/set para todos os campos que guardam listas
    # Exemplo para EMG Channel 1
    def get_emg_channel1_data(self):
        return json.loads(self.session_results_emg_channel1) if self.session_results_emg_channel1 else []

    def set_emg_channel1_data(self, data_list):
        self.session_results_emg_channel1 = json.dumps(data_list)