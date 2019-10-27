import tensorflow as tf

from data.npy_file_util import load_npy_file
from plot_util import print_loss_acc
from training.path_util import npy_files_dir_path, model_saving_dir_path


def execute():
    test_data = load_npy_file(npy_files_dir_path + 'test_z_data.npy')
    test_labels = load_npy_file(npy_files_dir_path + 'test_labels.npy')

    one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

    model = tf.keras.models.load_model(model_saving_dir_path + 'model_Z.h5')

    loss, acc = model.evaluate(test_data, one_hot_test_labels)
    print_loss_acc(loss, acc)


if __name__ == '__main__':
    execute()
