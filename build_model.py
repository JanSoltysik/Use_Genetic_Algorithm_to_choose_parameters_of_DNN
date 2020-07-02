import tensorflow as tf
from tensorflow import keras
import numpy as np

activation_functions = {
    0: keras.activations.sigmoid,
    1: keras.activations.tanh,
    2: keras.activations.relu,
    3: keras.activations.softmax,
    4: keras.activations.linear,
}


def get_output_layer_activation_function(nn_list) -> str:
    output_layer_activation_function = nn_list[-1][2]
    if output_layer_activation_function == 0:
        loss_fn = "binary_crossentropy"
    elif output_layer_activation_function == 3:
        loss_fn = "sparse_categorical_crossentropy"
    else:
        loss_fn = "mean_squared_error"

    return loss_fn


def build_layer(layer) -> keras.Layer:
    if layer[0] == 1:
        keras_layer = keras.layers.Dense(units=layer[1], activation=activation_functions[layer[2]],
                                         kernel_initializer="glorot_normal")
    elif layer[0] == 2:
        keras_layer = keras.layers.Conv2D(filters=layer[3], activation=activation_functions[layer[2]],
                                          kernel_size=(layer[4], layer[4]), strides=(layer[5], layer[5]),
                                          kernel_initializer="glorot_normal", padding="same")
    elif layer[0] == 3:
        keras_layer = keras.layers.MaxPool2D(pool_size=(layer[6], layer[6]), padding='same')
    elif layer[0] == 4:
        keras_layer = keras.layers.LSTM(units=layer[1], activation=activation_functions[layer[2]],
                                        kernel_initializer="glorot_normal")
    elif layer[0] == 5:
        keras_layer = keras.layers.Dropout(rate=layer[7])

    return keras_layer


def get_keras_model(nn_list, input_shape):
    model = keras.models.Sequential()

    if nn_list[0][0] == 1:
        model.add(keras.layers.Flatten())
        keras_layer = keras.layers.Dense(units=nn_list[0][1], activation=activation_functions[nn_list[0][2]],
                                         kernel_initializer="glorot_normal")
    elif nn_list[0][0] == 2:
        keras_layer = keras.layers.Conv2D(filters=nn_list[0][3],
                                          activation=activation_functions[nn_list[0][2]],
                                          kernel_size=(nn_list[0][4], nn_list[0][4]),
                                          strides=(nn_list[0][5], nn_list[0][5]),
                                          kernel_initializer="glorot_normal", padding="same",
                                          input_shape=input_shape)
    elif nn_list[0][0] == 4:
        keras_layer = keras.layers.LSTM(units=nn_list[0][1], activation=activation_functions[nn_list[0][2]],
                                        kernel_initializer="glorot_normal", input_shape=input_shape)
    model.add(keras_layer)

    prev_layer_type = nn_list[0][0]
    for layer in nn_list[1:]:
        curr_layer_type = layer[0]
        if curr_layer_type == 1 and prev_layer_type == 2:
            model.add(keras.layers.Flatten())
        model.add(build_layer(layer))

        # treating other layers as helping layers
        prev_layer_type = curr_layer_type if curr_layer_type in (1, 2, 4) else prev_layer_type

    return model


def partialy_train(genome, X, y, training_epochs, validation_split, verbose=0, final_train=False):
    nn_list = genome.genome
    input_shape = X.shape[1:]
    model = get_keras_model(nn_list, input_shape)

    loss_fn = get_output_layer_activation_function(nn_list)
    metric = "mse" if loss_fn is "mean_squared_error" else "accuracy"
    model.compile(loss=loss_fn, optimizer="adam", metrics=[metric])
    history = model.fit(X, y, epochs=training_epochs, validation_split=validation_split,
                        verbose=verbose)
    if final_train:
        return model
    else:
        return max(history.history["val_" + metric]), model.count_params()


def test_building_on_fashion_mnist():
    input, out = [2, 264, 2, 64, 3, 1, 2, 1], [1, 10, 3, 0, 0, 0, 0, 0]
    from nn_genome import NNGenome

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))

    gen = NNGenome(train_images.shape[1:], out, 0.1).genome
    print(gen)
    print(train_images.shape)
    print(partialy_train(gen, train_images, train_labels, 1, 0.4, 1))


if __name__ == '__main__':
    test_building_on_fashion_mnist()
