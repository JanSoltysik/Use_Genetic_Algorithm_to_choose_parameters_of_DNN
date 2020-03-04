import numpy as np
import copy
from tensorflow import keras
from nn_building_rules import LayerType, generate_array_for_layer, generate_value_for_nn_array_element,\
    ELEMENTS_FROM_ARRAY_USED_BY_LAYER, NN_STACKING_RULES, NNArrayStructure

K = keras.backend

class Genome:
    def __init__(self, ind_label, problem, stringModel, activation_functions, tModel=None, raw_score=0, raw_size=0):
        self.individual_label = ind_label
        self.problem = problem
        self.stringModel = stringModel
        self.activation_functions = activation_functions
        self.tModel = tModel
        self.raw_score = raw_score
        self.normalized_score = raw_score
        self.raw_size = raw_size
        self.checksum_vector = np.zeros(1)

    def partial_run(self, cross_validation_ratio=0.2, epochs=20, verbose=0, unroll=False, learning_rate_scheduler=None):
        self.tModel.load_data(verbose=1, cross_validation_ratio=cross_validation_ratio, unroll=unroll)
        self.tModel.epochs = epochs
        self.tModel.train_model(learning_rate_scheduler=learning_rate_scheduler, verbose=verbose)
        self.tModel.evaluate_model(cross_validation=True)

    def compute_raw_scores(self, epochs, cross_validation_ratio, verbose=0, unroll=False, learning_rate_scheduler=None):
        trainable_count = np.sum([K.count_params(param) for param in set(self.tModel.model.trainable_weights)],
                                 dtype=np.int)
        self.partial_run(cross_validation_ratio, epochs, verbose, unroll, learning_rate_scheduler)
        metric = self.tModel.scores['score_1']
        self.raw_score = metric
        self.normalized_score = metric

    def compute_fitnes(self, size_scaler):
        rounding_scaler = 10 ** 3
        if self.raw_size > rounding_scaler:
            trainable_count = round(self.raw_size / rounding_scaler) * rounding_scaler
        else:
            trainable_count = self.raw_size
        scaled_score = self.normalized_score

        if self.problem == 2:
            metric_score = (1 - scaled_score) * 10
        else:
            metric_score = scaled_score * 10

        size_score = np.log10(trainable_count)
        metric_scaler = 1 - size_scaler
        print(f"Metric scaler {metric_scaler}")
        print(f"Size scaler {size_scaler}")

        return metric_scaler * metric_score + size_scaler * size_score

    def compute_checksum_vector(self):
        self.checksum_vector = np.zeros(len(self.stringModel[0]))

        for layer in self.stringModel:
            layer_type = layer[0]
            self.checksum_vector[0] = self.checksum_vector[0] + layer_type.value

            useful_components = ELEMENTS_FROM_ARRAY_USED_BY_LAYER[layer_type]
            for index in useful_components:
                self.checksum_vector[index] = self.checksum_vector[index] + layer[index]


def generate_model(model=None, prev_component=LayerType.EMPTY, next_component=LayerType.EMPTY,
                   max_layers=64, more_layers_prob=0.7, activation_functions={}):
    layer_count = 0
    success = False
    model = [] if not model else model

    while True:
        curr_component = np.random.choice(NN_STACKING_RULES[prev_component])
        if curr_component == LayerType.PertrubateParam:
            NN_STACKING_RULES[LayerType.PertrubateParam] = NN_STACKING_RULES[prev_component]
        elif curr_component == LayerType.DROPOUT:
            NN_STACKING_RULES[LayerType.DROPOUT] = NN_STACKING_RULES[prev_component].copy()
            NN_STACKING_RULES[LayerType.DROPOUT].remove(LayerType.DROPOUT)

        rndm = np.random.random()
        rndm = 1 - np.sqrt(1 - rndm)

        if rndm > more_layers_prob:
            if next_component == LayerType.EMPTY or next_component in NN_STACKING_RULES[curr_component]
                layer = generate_array_for_layer(curr_component, activation_functions)
                model.append(layer)
                prev_component = curr_component
                success = True
                break
            elif max_layers >= layer_count:
                continue
            else:
                success = False
                model = []
                break
        else:
            layer = generate_array_for_layer(curr_component, activation_functions)
            model.append(layer)
            prev_component = curr_component

        return model, success

def initial_population(pop_size, problem_type, architecture_type, number_classes=2, more_layers_prob=0.7):
    population = []

    for i in range(pop_size):
            activation_functions = {}

            model_genotype, success = generate_model(more_layers_prob=more_layers_prob,
                                                    prev_component=architecture_type,
                                                     activation_functions=activation_functions)
            layer_first = generate_array_for_layer(architecture_type, activation_functions)

            if problem_type == 1:
                layer_last = [LayerType.FULLY_CONNECTED, 1, 4, 0, 0, 0, 0 ,0]
            else:
                layer_last = [LayerType.FULLY_CONNECTED, number_classes, 3, 0, 0, 0, 0, 0]

            model_genotype.append(layer_last)
            model_genotype = [layer_first] + model_genotype
            population.append(Genome(i, problem_type, model_genotype, activation_functions))

    return  population

def mutation(offsprings, mutation_ratio):
    for genome in offsprings:
        mutation_prob = np.random.random()
        if mutation_prob < mutation_ratio:
            len_model = len(genome.stringModel)
            random_layer_index = np.random.randint(len_model-1)
            genome.stringModel = mutate_layer(genome.stringModel, random_layer_index)

def mutate_layer(stringModel, layer_index):
    layer = stringModel[layer_index]
    next_layer = stringModel[layer_index + 1]
    layer_type = layer[0]
    next_layer_type = next_layer[0]

    characteristic = 0
    stringModelCopy = []

    if layer_type == LayerType.FULLY_CONNECTED:
        characteristic = np.random.choice([NNArrayStructure.NUMBER_OF_NEURONS.value,
                                           NNArrayStructure.ACTIVATION_FUNCTION.value,
                                           NNArrayStructure.DROPOUT_RATE.value])
    elif layer_type == LayerType.CONVOLUTIONAL:
        characteristic = np.random.choice([NNArrayStructure.ACTIVATION_FUNCTION.value,
                                           NNArrayStructure.NUMBERS_OF_FILTERS.value,
                                           NNArrayStructure.KERNEL_SIZE.value,
                                           NNArrayStructure.KERNEL_STRIDE.value,
                                           NNArrayStructure.DROPOUT_RATE.value])
    elif layer_type == LayerType.POOLING:
        characteristic = LayerType.POOLING.value
    elif layer_type == LayerType.DROPOUT:
        characteristic = LayerType.DROPOUT.value
    else:
        characteristic = -1

    characteristic = NNArrayStructure(characteristic)
    value = generate_value_for_nn_array_element(layer, characteristic)

    if characteristic != NNArrayStructure.DROPOUT_RATE and value != -1:
        layer[characteristic.value] = value
        if characteristic == NNArrayStructure.ACTIVATION_FUNCTION:
            activation = value
            rectify_activations_by_layer_type(stringModel, layer_type, activation)

    elif characteristic == NNArrayStructure.DROPOUT_RATE and value != -1:
        if layer_type != LayerType.DROPOUT:
            if next_layer_type == LayerType.DROPOUT:
                next_layer[NNArrayStructure.DROPOUT_RATE.value] = value
            else:
                stringModelCopy = stringModel[:layer_index+1]
                dropoutLayer = [LayerType.DROPOUT, 0, 0, 0, 0, 0, 0, 0]
                dropoutLayer[NNArrayStructure.DROPOUT_RATE.value] = value

                stringModelCopy.append(dropoutLayer)
                stringModelCopy.extend(stringModel[layer_index+1:])

                stringModel = copy.deepcopy(stringModelCopy)
        else:
            layer[NNArrayStructure.DROPOUT_RATE.value] = value

    return  stringModel

def rectify_activations_by_layer_type(stringModel, layer_type, activation):
    for i in range(len(stringModel)-1):
        layer = stringModel[i]
        if layer[0] == layer_type:
            layer[2] = activation

def rectify_activation_offspring(stringModel):
    activation_functions = {}
    for i in range(len(stringModel)-1):
        layer = stringModel[i]
        layer_type = layer[0]
        activation = layer[2]

        if layer_type in activation_functions:
            layer[2] = activation_functions[layer_type]
        else:
            activation_functions[layer_type] = activation

    return activation_functions

def tournament_selection(subpopulation):
    if len(subpopulation) < 2:
        print("At least are required")
        return None
    else:
        most_fit = subpopulation[0]

    for index in range(1, len(subpopulation)):
        individual = subpopulation[index]
        if individual.fitness < most_fit.fitness:
            most_fit = individual

    return most_fit

def binary_tournament_selection(population):
    parent_pool = []
    for index in range(len(population) // 2):
        if population[index * 2].fitness < population[index * 2 + 1].fitness:
            parent_pool.append(population[index*2])
        else:
            parent_pool.append((population[index * 2 + 1]))

    return  parent_pool

def population_crossover(parent_pool, max_layers=3):
    pop_size = len(parent_pool) // 2
    problem_type = parent_pool[0].problem_type
    offsprings = []
    i  = 0

    for index in range(pop_size):
        parent_1 = parent_pool[index * 2]
        parent_2 = parent_pool[index * 2 + 1]

        point11, point12, point21, point22, success = two_point_crossover(parent_1, parent_2, max_layers)

        if success:
            offsprings_stringModel = parent_1.stringModel[:point11]
            offsprings_stringModel.extend(parent_2.stringModel[point21:point22 + 1])
            offsprings_stringModel.extend(parent_1.stringModel[point12+1:])
            offsprings_stringModel = copy.deepcopy(offsprings_stringModel)

            activation_functions = rectify_activation_offspring(offsprings_stringModel)
            offspring = Genome(pop_size + i, problem_type, offsprings_stringModel, activation_functions)
            offsprings.append(offspring)
            i = i + 1

    return offsprings

def two_point_crossover(parent1, parent2, max_layers, max_attempts=5):
    stringModel1 = parent1.stringModel
    len_model1 = len(stringModel1) - 1

    attempts = 0
    success = False

    while attempts < max_attempts:
        temp = 0
        attempts = attempts + 1
        compatible_substructures = []

        point11 = np.random.randint(len_model1)
        point12 = np.random.randint(len_model1)
        point21 = point22 = -1

        if point11 > point12:
            point11, point12 = point12, point11
        elif point11 == point12:
            point12 = len_model1 -1

        first_layer = True if point11 == 0 else False

        if first_layer:
            layer_prev = stringModel1[point11]
        else:
            layer_prev = stringModel1[point11 - 1]
        layer_next = stringModel1[point12 + 1]

        compatible_previous, compatible_next = find_match(parent2, layer_prev, layer_next, first_layer, max_layers)

        if not compatible_next or not compatible_previous:
            for i in compatible_previous:
                for j in compatible_next:
                    if 0 <= j - i < max_layers:
                        compatible_substructures.append((i, j))


            if not compatible_substructures:
                k = np.random.randint(len(compatible_substructures))
                chosen_substructure = compatible_substructures[k]
                point21 = chosen_substructure[0]
                point22 = chosen_substructure[1]
                success = True
                break

    return point11, point12, point21, point22, success

def find_match(parent, layer_prev, layer_next, first_layer, max_layers):
    stringModel = parent.stringModel
    len_model = len(stringModel)
    point11 = 0
    point12 = 0
    compatible_previous = []
    compatible_next = []

    for i in range(len_model - 1):
        layer = stringModel[i]
        if first_layer:
            if layer[0] == layer_prev[0]:
                compatible_previous.append(i)
        else:
            compatible_layers = NN_STACKING_RULES[layer_prev[0]]
            if layer[0] in compatible_layers:
                compatible_previous.append(i)

        compatible_layers = NN_STACKING_RULES[layer[0]]

        if layer_next[0] in compatible_layers:
            compatible_next.append(i)

    return compatible_previous, compatible_next


def generation_similar(population, max_similar, similar_threshold=0.9):
    new_pop = []
    len_pop = len(population)
    pairs = []
    distances = {}
    max_distance = 0
    max_pair = None
    similar = 0
    generation_similar = False

    for i in range(len_pop):
        for j in range(len_pop):
            if j > i:
                pairs.append((i, j))

    for i, j in pairs:
        distance_norm = distance_between_models(population[i].stringModel, population[j].stringModel)
        distances[(i, j)] = distance_norm

        if distance_norm > max_distance:
            max_distance = distance_norm
            max_pair = (i, j)

    if max_distance == 0:
        generation_similar = True
    else:
        for key in distances:
            normalized_distance = distances[key] / max_distance
            distances[key] = normalized_distance

            if normalized_distance < similar_threshold and key != max_pair:
                similar = similar + 1
    if similar > max_similar:
        generation_similar = True

    return generation_similar

def distance_between_models(stringModel1, stringModel2):
    len_model1 = len(stringModel1)
    len_model2 = len(stringModel2)

    len_layer = len(stringModel1[0])
    layer_distance = np.zeros(len_layer)

    distance = 0

    if len_model1 > len_model2:
        for i in range(len_model2 - 1):
            layer_distance[0] = stringModel1[i][0].value - stringModel2[i][0].value

            for j in range(1, len_layer):
                layer_distance[j] = stringModel1[i][j] - stringModel2[i][j]

            distance += np.linalg.norm(layer_distance, 2)

        for i in range(len_model2 - 1, len_model1 - 1):
            layer_distance[0] - stringModel1[i][0].value

            for j in range(1, len_layer):
                layer_distance[j] = stringModel1[i][j]

            distance += np.linalg.norm(layer_distance, 2)

    else:
        for i in range(len_model1 - 1):
            layer_distance[0] = stringModel2[i][0].value - stringModel1[i][0].value

            for j in range(1, len_layer):
                layer_distance[j] = stringModel2[i][j] - stringModel1[i][j]

            distance += np.linalg.norm(layer_distance, 2)

        for i in range(len_model1 - 1, len_model2 - 1):
            layer_distance[0] = stringModel2[i][0].value
            for j in range(1, len_layer):
                layer_distance[j] = stringModel2[i][j]

            distance += np.linalg.norm(layer_distance, 2)

    return distance



