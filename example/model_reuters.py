import numpy as np
import tensorflow as tf

from plot_util import show_loss, show_accuracy


def execute():
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.reuters.load_data(num_words=10000)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
    one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(46, activation='softmax')
    ])

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val)
    )
    show_loss(history)
    show_accuracy(history)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


if __name__ == '__main__':
    execute()
