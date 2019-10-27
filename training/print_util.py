import numpy as np


def print_predictions(model, test_data):
    indices_to_check = np.asarray([50, 100, 150, 200, 250, 300])
    predictions = model.predict_classes(test_data)
    for it in indices_to_check:
        print(predictions[it])
