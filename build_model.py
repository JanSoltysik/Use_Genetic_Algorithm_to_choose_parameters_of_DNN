import tensorflow as tf
from tensorflow import keras

activation_functions = {
    0: keras.activations.sigmoid,
    1: keras.activations.tanh,
    2: keras.activations.relu,
    3: keras.activations.softmax,
    4: keras.activations.linear,
}


def build_layer(layer):
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
        keras_layer = keras.layers.Dense(units=nn_list[0][1], activation=activation_functions[nn_list[0][2]],
                                         kernel_initializer="glorot_normal", input_shape=input_shape)
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


def main():
    input, out = [2, 264, 2, 64, 3, 1, 2, 1], [1, 10, 3, 0, 0, 0, 0, 0]
    from nn_genome import NNGenome
    gen = NNGenome(input, out, 1).genome
    print(gen)
    get_keras_model(gen, input_shape=[28, 28, 1])

main()