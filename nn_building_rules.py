import numpy as np
import enum


class LayerType(enum.Enum):
    FULLY_CONNECTED = 1
    CONVOLUTIONAL = 2
    POOLING = 3
    RECURRENT = 4
    DROPOUT = 5
    PertrubateParam = 6
    EMPTY = 7


class ActivationFunction(enum.Enum):
    SIGMOID = 1
    HYPERBOLIC_TANGENT = 2
    RELU = 3
    SOFTMAX = 4
    LINEAR = 5


class NNArrayStructure(enum.Enum):
    LAYER_TYPE = 1
    NUMBER_OF_NEURONS = 2
    ACTIVATION_FUNCTION = 3
    NUMBERS_OF_FILTERS = 4
    KERNEL_SIZE = 5
    KERNEL_STRIDE = 6
    POOLING_SIZE = 7
    DROPOUT_RATE = 8


MAX_VALUE_FOR_ARRAY_ELEMENT = {
    NNArrayStructure.LAYER_TYPE: 5,
    NNArrayStructure.NUMBER_OF_NEURONS: 1024 / 8,  # final value equals (8 * picked number)
    NNArrayStructure.ACTIVATION_FUNCTION: 4,
    NNArrayStructure.NUMBERS_OF_FILTERS: 512 / 8,  # same as above
    NNArrayStructure.KERNEL_SIZE: 6,  # final value equals 3^(picked number)
    NNArrayStructure.KERNEL_STRIDE: 6,
    NNArrayStructure.POOLING_SIZE: 6,  # final value equals 2^(picked number)
    NNArrayStructure.DROPOUT_RATE: 0.7,
}

NN_STACKING_RULES = {
    LayerType.FULLY_CONNECTED: (LayerType.FULLY_CONNECTED, LayerType.DROPOUT),
    LayerType.CONVOLUTIONAL: (LayerType.FULLY_CONNECTED, LayerType.CONVOLUTIONAL, LayerType.POOLING,
                              LayerType.RECURRENT, LayerType.DROPOUT),
    LayerType.POOLING: (LayerType.FULLY_CONNECTED, LayerType.CONVOLUTIONAL),
    LayerType.RECURRENT: (LayerType.FULLY_CONNECTED, LayerType.RECURRENT),
    LayerType.DROPOUT: (LayerType.FULLY_CONNECTED, LayerType.CONVOLUTIONAL, LayerType.RECURRENT),
    LayerType.PertrubateParam: tuple(),
    LayerType.EMPTY: (LayerType.FULLY_CONNECTED, LayerType.RECURRENT, LayerType.CONVOLUTIONAL),
}

ACTIVATION_FOR_LAYER_TYPE = {
    LayerType.FULLY_CONNECTED: (ActivationFunction.SIGMOID, ActivationFunction.HYPERBOLIC_TANGENT,
                                ActivationFunction.RELU),
    LayerType.CONVOLUTIONAL: (ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU),
    LayerType.RECURRENT: (ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU),
}

ELEMENTS_FROM_ARRAY_USED_BY_LAYER = {
    LayerType.FULLY_CONNECTED: (NNArrayStructure.NUMBER_OF_NEURONS, NNArrayStructure.ACTIVATION_FUNCTION),
    LayerType.CONVOLUTIONAL: (NNArrayStructure(i) for i in range(2, 6)),
    LayerType.POOLING: (NNArrayStructure.POOLING_SIZE,),
    LayerType.RECURRENT: (NNArrayStructure.NUMBER_OF_NEURONS, NNArrayStructure.ACTIVATION_FUNCTION),
    LayerType.DROPOUT: (NNArrayStructure.DROPOUT_RATE,),
}


def increase_dropout_rate(ratio):
    """
    ratio in range [0, 1]
    then returns increased ratio by 0.5
    """
    ratio *= 100
    first_digit_from_ratio = ratio // 10
    second_digit_from_ratio = ratio % 10
    second_digit_in_new_ratio = 5 if second_digit_from_ratio <= 5 else 0

    return first_digit_from_ratio / 10 + second_digit_in_new_ratio / 100


def generate_number_of_neurons():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.NUMBER_OF_NEURONS]
    return 8 * np.random.randint(1, max_value + 1)


def generate_activation_function(layer_type):
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.ACTIVATION_FUNCTION]
    return np.random.randint(1, max_value) if layer_type != LayerType.RECURRENT else 2


def generate_number_of_filters():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.NUMBERS_OF_FILTERS]
    return 8 * np.random.randint(1, max_value + 1)


def generate_kernel_size():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.KERNEL_SIZE]
    return 3 ** np.random.randint(1, max_value + 1)


def generate_kernel_stride():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.KERNEL_STRIDE]
    return np.random.randint(1, max_value + 1)


def generate_pooling_size():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.POOLING_SIZE]
    return 2 ** np.random.randint(1, max_value + 1)


def generate_dropout_rate():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.DROPOUT_RATE]
    return round(np.random.uniform(0.1, max_value), 2)


# delete layer parametr or split this method?
def generate_value_for_nn_array_element(layer, elementType):
    layer_type = layer[0]
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT(elementType)

    if elementType == NNArrayStructure.NUMBER_OF_NEURONS:
        return generate_number_of_neurons()
    elif elementType == NNArrayStructure.ACTIVATION_FUNCTION:
        return generate_activation_function(layer_type)
    elif elementType == NNArrayStructure.NUMBERS_OF_FILTERS:
        return generate_number_of_filters()
    elif elementType == NNArrayStructure.KERNEL_SIZE:
        return generate_kernel_size()
    elif elementType == NNArrayStructure.KERNEL_STRIDE:
        return generate_kernel_stride()
    elif elementType == NNArrayStructure.POOLING_SIZE:
        return generate_pooling_size()
    elif elementType == NNArrayStructure.DROPOUT_RATE:
        return generate_dropout_rate()
    else:
        return -1


def generate_array_for_layer(layer_type, activation_functions={}):
    """
    activation_functions = {LayerType : ActivationFunction}
    """
    new_layer = [layer_type] + [0] * 7

    if layer_type == LayerType.FULLY_CONNECTED or layer_type == LayerType.RECURRENT:
        new_layer[1] = generate_number_of_neurons()

    if layer_type in (LayerType.FULLY_CONNECTED, LayerType.RECURRENT, LayerType.CONVOLUTIONAL):
        if layer_type in activation_functions:
            new_layer[2] = activation_functions[layer_type]
        else:
            new_layer[2] = generate_value_for_nn_array_element(new_layer,
                                                               NNArrayStructure.ACTIVATION_FUNCTION)

    if layer_type == LayerType.CONVOLUTIONAL:
        new_layer[3] = generate_number_of_filters()
        new_layer[4] = generate_kernel_size()
        new_layer[5] = generate_kernel_stride()

    if layer_type == LayerType.POOLING:
        new_layer[6] = generate_pooling_size()

    if layer_type == LayerType.DROPOUT:
        new_layer[7] = generate_dropout_rate()

    return new_layer
