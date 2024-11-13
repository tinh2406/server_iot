import numpy as np
from tensorflow import keras
import neurokit2 as nk

load_model = keras.models.load_model(f'my_model.keras')



# Sample frequency and loading signal data
sr = 125  # sample frequency
MIN_DIFF = 80  # Minimum distance for peak adjustment
mode = 160  # Mode for peak adjustment

mapping = {
    0: 'N',
    1: 'S',
    2: 'V',
    3: 'F',
    4: 'Q'
}

def predict(values):
    """
    :param values list[float]: list of values
    :return: label
    """
    values = values/np.max(values)

    _, rpeaks = nk.ecg_peaks(values, sampling_rate=sr)
    rpeaks = rpeaks['ECG_R_Peaks'].astype(int)

    mode = 100

    cycles = []
    for peak in rpeaks:
        left, right = peak, peak + mode + 4
        if np.all([left > 0, right < len(values)]):
            cycles.append(values[left:right])

    data = np.array(cycles)
    data = np.pad(data, ((0, 0), (0, 187 - data.shape[1])), 'constant', constant_values=0)

    prediction = load_model.predict(data)

    indexs = np.argmax(prediction, axis=1)
    
    unique, counts = np.unique(indexs, return_counts=True)
    most_frequent_label = unique[np.argmax(counts)]

    return mapping[most_frequent_label], [mapping[i] for i in indexs]

