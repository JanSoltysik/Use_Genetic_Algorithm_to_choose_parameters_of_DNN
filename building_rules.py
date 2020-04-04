import enum


class LayerType(enum.Enum):
    FULLY_CONNECTED = 1
    CONVOLUTIONAL = 2
    POOLING = 3
    DROPOUT = 5


class ActivationFunction(enum.Enum):
    SIGMOID = 0
    HYPERBOLIC_TANGENT = 1
    RELU = 2
    SOFTMAX = 3
    LINEAR = 4


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
    NNArrayStructure.NUMBER_OF_NEURONS: 50,  # final value equals (8 * picked number)
    NNArrayStructure.ACTIVATION_FUNCTION: 4,
    NNArrayStructure.NUMBERS_OF_FILTERS: 10,  # same as above
    NNArrayStructure.KERNEL_SIZE: 2,  # final value equals 3^(picked number)
    NNArrayStructure.KERNEL_STRIDE: 3,
    NNArrayStructure.POOLING_SIZE: 3,  # final value equals 2^(picked number)
    NNArrayStructure.DROPOUT_RATE: 0.5,
}

NN_STACKING_RULES = {
    LayerType.FULLY_CONNECTED: (LayerType.FULLY_CONNECTED, LayerType.DROPOUT),
    LayerType.CONVOLUTIONAL: (LayerType.FULLY_CONNECTED, LayerType.CONVOLUTIONAL, LayerType.POOLING,
                              LayerType.DROPOUT),
    LayerType.POOLING: (LayerType.FULLY_CONNECTED, LayerType.CONVOLUTIONAL),
    LayerType.DROPOUT: (LayerType.FULLY_CONNECTED, LayerType.CONVOLUTIONAL),
}

ACTIVATION_FOR_LAYER_TYPE = {
    LayerType.FULLY_CONNECTED: (ActivationFunction.SIGMOID, ActivationFunction.HYPERBOLIC_TANGENT,
                                ActivationFunction.RELU),
    LayerType.CONVOLUTIONAL: (ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU),
}

ELEMENTS_FROM_ARRAY_USED_BY_LAYER = {
    LayerType.FULLY_CONNECTED: (NNArrayStructure.NUMBER_OF_NEURONS, NNArrayStructure.ACTIVATION_FUNCTION),
    LayerType.CONVOLUTIONAL: tuple(NNArrayStructure(i) for i in range(3, 7)),
    LayerType.POOLING: (NNArrayStructure.POOLING_SIZE,),
    LayerType.DROPOUT: (NNArrayStructure.DROPOUT_RATE,),
}
