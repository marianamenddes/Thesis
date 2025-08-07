import numpy as np
from scipy.signal import butter, filtfilt

def preprocess_emg_signal(emg_data, fs=1500):
    """
    Aplica Demean + filtro passa-banda 30–300 Hz ao sinal EMG.
    """
    emg_data = np.array(emg_data)
    emg_data = emg_data - np.mean(emg_data)

    nyq = 0.5 * fs
    low = 30 / nyq
    high = 300 / nyq

    b, a = butter(N=4, Wn=[low, high], btype='band')
    filtered = filtfilt(b, a, emg_data)

    return filtered.tolist()
