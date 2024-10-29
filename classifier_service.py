import numpy as np
from tensorflow import keras

load_model = keras.models.load_model(f'my_model.keras')

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
    data = np.array(values).reshape(1, 187, 1)
    index = np.argmax(load_model.predict(data))

    return mapping[index]