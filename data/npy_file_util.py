import os

import numpy as np


def load_npy_file(file_name):
    try:
        data = np.load(file_name, allow_pickle=True)
    except IOError:
        data = np.zeros((0, 0), dtype=np.float)
    return data


def save_npy_file(data, file_name):
    np.save(file_name, data, allow_pickle=True)


def create_dir_if_necessary(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


num_data_points = 200
