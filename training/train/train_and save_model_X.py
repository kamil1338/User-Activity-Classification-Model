import tensorflow as tf

from data.npy_file_util import load_npy_file, create_dir_if_necessary
from plot_util import show_loss, show_accuracy, print_loss_acc
from training.path_util import npy_files_dir_path, model_saving_dir_path


def execute():
    train_data = load_npy_file(npy_files_dir_path + 'train_x_data.npy')
    valid_data = load_npy_file(npy_files_dir_path + 'valid_x_data.npy')
    test_data = load_npy_file(npy_files_dir_path + 'test_x_data.npy')

    train_labels = load_npy_file(npy_files_dir_path + 'train_labels.npy')
    valid_labels = load_npy_file(npy_files_dir_path + 'valid_labels.npy')
    test_labels = load_npy_file(npy_files_dir_path + 'test_labels.npy')

    one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
    one_hot_valid_labels = tf.keras.utils.to_categorical(valid_labels)
    one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        one_hot_train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(valid_data, one_hot_valid_labels)
    )

    create_dir_if_necessary(model_saving_dir_path)

    model.save(model_saving_dir_path + 'model_X.h5')

    loss, acc = model.evaluate(test_data, one_hot_test_labels)
    print_loss_acc(loss, acc)

    show_loss(history)
    show_accuracy(history)


if __name__ == '__main__':
    execute()
