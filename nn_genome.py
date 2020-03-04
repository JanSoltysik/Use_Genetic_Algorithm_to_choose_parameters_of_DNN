import building_rules
from building_rules import MAX_VALUE_FOR_ARRAY_ELEMENT, NNArrayStructure, LayerType, \
    NN_STACKING_RULES
import random
import numpy as np


def generate_number_of_neurons():
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.NUMBER_OF_NEURONS]
    return 8 * random.randint(1, max_value)


def generate_activation_function(layer_type):
    max_value = MAX_VALUE_FOR_ARRAY_ELEMENT[NNArrayStructure.ACTIVATION_FUNCTION]
    return random.randint(1, max_value - 1) if layer_type != LayerType.RECURRENT else 2


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


def generate_layer(layer_type, activation_function=0):
    indices_of_allowed_elements = building_rules.ELEMENTS_FROM_ARRAY_USED_BY_LAYER[LayerType(layer_type)]
    layer = [0] * 8
    layer[0] = layer_type
    if NNArrayStructure.ACTIVATION_FUNCTION in indices_of_allowed_elements:
        if activation_function:
            layer[2] = activation_function
        else:
            layer[2] = generate_activation_function()
    else:
        layer[2] = 0

    indices = (1, 3, 4, 5, 6, 7)
    functions = (generate_number_of_neurons, generate_number_of_filters, generate_pooling_size,
                 generate_kernel_stride, generate_pooling_size, generate_dropout_rate)

    for ind, generate in zip(indices, functions):
        layer[ind] = generate() \
            if NNArrayStructure(ind+1) in indices_of_allowed_elements else 0

    return layer


class NNGenome:
    def __init__(self, input_layer, output_layer, more_layers_probability, max_layers=10):
        self.more_layers_probability = more_layers_probability
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.max_layers = max_layers

        self.genome = [self.input_layer, *self.generateHiddenLayers(), self.output_layer]

    def generateHiddenLayers(self):
        nn_type = LayerType(self.input_layer[0])
        activation_function = self.input_layer[2]
        possible_layers = NN_STACKING_RULES[nn_type]

        hidden_layers = []
        added_layers = 0
        prev_layer_type = nn_type.value
        while added_layers < self.max_layers:
            add_next_layer_prob = 1 - np.sqrt(1 - random.random())

            current_layer_type = random.choice(
                list(set(NN_STACKING_RULES[LayerType(prev_layer_type)]) & set(NN_STACKING_RULES[nn_type]))).value
            current_layer = generate_layer(current_layer_type, activation_function)
            prev_layer_type = current_layer[0]
            hidden_layers.append(current_layer)

            if add_next_layer_prob > self.more_layers_probability:
                break

            added_layers += 1

        return hidden_layers

    def getFitness(self):
        pass  # potrzebne powiazanie z kerasem
