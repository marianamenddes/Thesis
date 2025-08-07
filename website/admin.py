from django.contrib import admin
from .models import Doctor, Patient, Sessions
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.urls import path
from django.contrib.auth.admin import UserAdmin
# Register your models here.


@admin.register(Doctor) # Forma moderna de registar o modelo
class DoctorAdmin(UserAdmin): # <<--- AGORA HERDA DE UserAdmin
    # Os campos name e id_number que definiste no teu modelo Doctor podem ser usados
    list_display = ("name", "id_number", "email", "is_staff", "is_active") # Adicionei email, is_staff, is_active para melhor gestão

    # Campos que estarão disponíveis para edição nos formulários do admin
    # Vais precisar de adicionar os teus campos personalizados aos fieldsets padrão do UserAdmin
    fieldsets = UserAdmin.fieldsets + (
        (None, {'fields': ('name', 'id_number', 'phone_number')}), # Adiciona os teus campos personalizados aqui
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        (None, {'fields': ('name', 'id_number', 'phone_number')}),
    )
    # Podes adicionar search_fields, list_filter, etc. aqui como farias com UserAdmin


class PatientAdmin(admin.ModelAdmin):
    list_display = ("name", "patient_number","sex", "age", "dominant_leg_side")

class SessionsAdmin(admin.ModelAdmin):
    list_display = (
       "session_id",
       "Patient",
       "session_type", # Novo campo
       "session_start_time", # Bom para visualização
       "session_time", # Novo campo (duração)
       "session_status_r",
       "session_status_l",
       "session_status_emg", # Novo campo
       "emg_measured_leg", # Novo campo
       "emg_prediction", # Novo campo
   )
    list_filter = (
        "session_type",
        "session_status_r",
        "session_status_l",
        "session_status_emg",
        "emg_measured_leg",
        "Patient", # Pode filtrar por paciente
    )
    fieldsets = (
        (None, {
            'fields': ('Patient', 'session_id', 'notes')
        }),
        ('Detalhes da Sessão', {
            'fields': (
                'session_type',
                'session_start_time',
                'session_time',
                'emg_measured_leg',
                'emg_prediction',
            )
        }),
        ('Status de Coleta', {
            'fields': (
                'session_status_r',
                'session_status_l',
                'session_status_emg',
            )
        }),
        ('Dados Coletados (JSON)', {
            'classes': ('collapse',), # Opcional: faz com que esta seção seja recolhível
            'fields': (
                'session_results_gait_right',
                'session_results_gait_left',
                'session_results_emg_channel1',
                'session_results_emg_channel2',
                'session_results_emg_channel3',
            )
        }),
    )

    # Campos que serão apenas de leitura na página de detalhes (impedindo edição acidental)
    # É uma boa ideia para os resultados de dados e campos calculados.
    readonly_fields = (
        "session_id",
        "session_start_time",
        "session_time",
        "session_results_gait_right",
        "session_results_gait_left",
        "session_results_emg_channel1",
        "session_results_emg_channel2",
        "session_results_emg_channel3",
        "emg_prediction", # Se a predição é automática
    )


admin.site.register(Patient, PatientAdmin)
admin.site.register(Sessions, SessionsAdmin)



