import datetime
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing

from data.generate_npy_files_util import npy_files_saving_dir_path, raw_data_path
from data.npy_file_util import save_npy_file


# X data structure: (recording, x_data)
# Y data structure: (recording, y_data)
# Z data structure: (recording, z_data)
# Labels structure: (activity)

def execute():
    general_data_frame = pd.read_csv(raw_data_path)

    unique_recording_ids = get_unique_recording_ids(general_data_frame)
    np.random.shuffle(unique_recording_ids)

    os.mkdir(npy_files_saving_dir_path)

    x_data, y_data, z_data, labels = get_recording_data(
        general_data_frame,
        unique_recording_ids,
        num_data_points=200
    )

    encoded_labels, classes = get_encoded_labels(labels)

    split_generate_files(x_data, y_data, z_data, encoded_labels, classes)


def get_unique_recording_ids(general_data_frame):
    unique_recordings = general_data_frame['recording_id'].unique()
    return unique_recordings


def get_recording_data(general_data_frame, unique_recording_ids, num_data_points):
    x_data = np.zeros((len(unique_recording_ids), num_data_points))
    y_data = np.zeros((len(unique_recording_ids), num_data_points))
    z_data = np.zeros((len(unique_recording_ids), num_data_points))
    labels = []
    for idx, unique_id in enumerate(unique_recording_ids):
        sorted_data_frame = get_sorted_data_frame_for_recording_id(unique_id, general_data_frame)

        x_parameter_data_slice = sorted_data_frame['x_data'].values
        y_parameter_data_slice = sorted_data_frame['y_data'].values
        z_parameter_data_slice = sorted_data_frame['z_data'].values
        activity_slice = sorted_data_frame['activity'].unique()

        insert_row_to_data_array(idx, x_data, x_parameter_data_slice)
        insert_row_to_data_array(idx, y_data, y_parameter_data_slice)
        insert_row_to_data_array(idx, z_data, z_parameter_data_slice)
        labels.append(activity_slice[0])
    return x_data, y_data, z_data, labels


def get_sorted_data_frame_for_recording_id(recording_id, general_data_frame):
    recording_data = general_data_frame[general_data_frame['recording_id'] == recording_id]
    recording_data = recording_data.sort_values('time', ascending=True)
    return recording_data


def insert_row_to_data_array(idx, data_array, row):
    end = min(data_array.shape[1], len(row))
    for i in range(0, end):
        data_array[idx][i] = row[i]


def get_encoded_labels(labels):
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    classes = np.asarray(encoder.classes_)
    return encoded_labels, classes


def split_generate_files(x_data, y_data, z_data, encoded_labels, classes):
    train_x_data, test_x_data, valid_x_data = split_train_test_valid(x_data)
    save_npy_file(train_x_data, npy_files_saving_dir_path + 'train_x_data')
    save_npy_file(test_x_data, npy_files_saving_dir_path + 'test_x_data')
    save_npy_file(valid_x_data, npy_files_saving_dir_path + 'valid_x_data')

    train_y_data, test_y_data, valid_y_data = split_train_test_valid(y_data)
    save_npy_file(train_y_data, npy_files_saving_dir_path + 'train_y_data')
    save_npy_file(test_y_data, npy_files_saving_dir_path + 'test_y_data')
    save_npy_file(valid_y_data, npy_files_saving_dir_path + 'valid_y_data')

    train_z_data, test_z_data, valid_z_data = split_train_test_valid(z_data)
    save_npy_file(train_z_data, npy_files_saving_dir_path + 'train_z_data')
    save_npy_file(test_z_data, npy_files_saving_dir_path + 'test_z_data')
    save_npy_file(valid_z_data, npy_files_saving_dir_path + 'valid_z_data')

    train_labels, test_labels, valid_labels = split_train_test_valid(encoded_labels)
    save_npy_file(train_labels, npy_files_saving_dir_path + 'train_labels')
    save_npy_file(test_labels, npy_files_saving_dir_path + 'test_labels')
    save_npy_file(valid_labels, npy_files_saving_dir_path + 'valid_labels')

    save_classes_to_txt_file(classes, npy_files_saving_dir_path + 'classes.txt')


def split_train_test_valid(data):
    train_data_percentage = 0.50
    test_data_percentage = 0.25

    train_count = int(len(data) * train_data_percentage)
    test_count = int(len(data) * test_data_percentage)

    train_set = data[:train_count]
    test_set = data[train_count:train_count + test_count]
    valid_set = data[train_count + test_count:]
    return train_set, test_set, valid_set


def save_classes_to_txt_file(data, file_name):
    classes_df = pd.DataFrame(data)
    classes_df.to_csv(file_name)

if __name__ == '__main__':
    print('START:', datetime.datetime.now())
    execute()
    print('END:', datetime.datetime.now())
