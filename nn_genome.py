import random
import numpy as np
import building_rules
# import build_model
from building_rules import MAX_VALUE_FOR_ARRAY_ELEMENT, NNArrayStructure, LayerType, \
    NN_STACKING_RULES, ELEMENTS_FROM_ARRAY_USED_BY_LAYER


def generate_number_of_neurons():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.NUMBER_OF_NEURONS]
    return 8 * random.randint(1, max_value)


def generate_activation_function():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.ACTIVATION_FUNCTION]
    return random.randint(0, max_value - 2)


def generate_number_of_filters():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.NUMBERS_OF_FILTERS]
    return 8 * random.randint(1, max_value)


def generate_kernel_size():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.KERNEL_SIZE]
    return 3 ** random.randint(1, max_value)


def generate_kernel_stride():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.KERNEL_STRIDE]
    return random.randint(1, max_value)


def generate_pooling_size():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.POOLING_SIZE]
    return 2 ** random.randint(1, max_value)


def generate_dropout_rate():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.DROPOUT_RATE]
    return round(random.uniform(0.1, max_value), 2)


def generate_layer(layer_type, activation_function=None):
    indices_of_allowed_elements = building_rules.ELEMENTS_FROM_ARRAY_USED_BY_LAYER[LayerType(layer_type)]
    layer = [0] * 8
    layer[0] = layer_type
    if NNArrayStructure.ACTIVATION_FUNCTION in indices_of_allowed_elements:
        if activation_function is not None:
            layer[2] = activation_function
        else:
            layer[2] = generate_activation_function()
    else:
        layer[2] = 0

    indices = (1, 3, 4, 5, 6, 7)
    functions = (generate_number_of_neurons, generate_number_of_filters, generate_kernel_size,
                 generate_kernel_stride, generate_pooling_size, generate_dropout_rate)

    for ind, generate in zip(indices, functions):
        layer[ind] = generate() \
            if NNArrayStructure(ind + 1) in indices_of_allowed_elements else 0

    return layer


generate_layer_elem = {
    2: generate_number_of_neurons,
    3: generate_activation_function,
    4: generate_number_of_filters,
    5: generate_kernel_size,
    6: generate_kernel_stride,
    7: generate_pooling_size,
    8: generate_dropout_rate,
}


class NNGenome:
    def __init__(self, input_shape, output_layer, architecture_type,
                 more_layers_probability,
                 max_layers=10, genome=None):
        self.more_layers_probability = more_layers_probability
        self.input_shape = input_shape
        self.architecture_type = architecture_type
        self.output_layer = output_layer
        self.max_layers = max_layers

        self.input_layer = self.generateInputLayer()
        if not genome:
            self.genome = [self.input_layer, *self.generateHiddenLayers(), self.output_layer]
        else:
            self.genome = genome
            self.input_layer = genome[0]

    def generateInputLayer(self):
        if self.architecture_type == 1:
            layer_type = np.random.choice((LayerType.FULLY_CONNECTED, LayerType.CONVOLUTIONAL)).value
        else:
            layer_type = LayerType.FULLY_CONNECTED.value
        layer = generate_layer(layer_type)
        if self.architecture_type == 1 and layer_type == 1:
            layer[1] = np.prod(self.input_shape)
        return layer

    def generateHiddenLayers(self):
        nn_type = LayerType(self.input_layer[0])
        activation_function = self.input_layer[2]

        hidden_layers = []
        added_layers = 0
        prev_layer_type = nn_type.value

        while added_layers < self.max_layers:
            add_next_layer_prob = 1 - np.sqrt(1 - random.random())

            # After we switch from convolutional to fully connected we can't add any more conv layers
            if prev_layer_type == 1:
                nn_type = LayerType.FULLY_CONNECTED

            available_layers = list(set(NN_STACKING_RULES[LayerType(prev_layer_type)]) & \
                                    set(NN_STACKING_RULES[nn_type]))
            current_layer_type = random.choice(available_layers).value
            current_layer = generate_layer(current_layer_type, activation_function)
            prev_layer_type = current_layer[0]
            hidden_layers.append(current_layer)

            if add_next_layer_prob > self.more_layers_probability:
                break

            added_layers += 1
        return hidden_layers

    def getActivationFunction(self):
        for layer in self.genome:
            if NNArrayStructure.ACTIVATION_FUNCTION in ELEMENTS_FROM_ARRAY_USED_BY_LAYER[LayerType(layer[0])]:
                return layer[NNArrayStructure.ACTIVATION_FUNCTION.value]

    def __str__(self):
        return str(self.genome)

    def __repr__(self):
        return self.__str__()
