import copy
import numpy as np
import nn_genome
import build_model
from building_rules import LayerType, ActivationFunction, NN_STACKING_RULES, \
    NNArrayStructure, ELEMENTS_FROM_ARRAY_USED_BY_LAYER


def replace_last_three_digits_with_zeros(number):
    round_to = 10 ** 3
    if number <= round_to:
        return number

    return round(number / round_to) * round_to


def rescale_size_of_nn(size):
    scaled_size = replace_last_three_digits_with_zeros(size)
    return np.log10(scaled_size)


def rectify_activation_function_nn(nn_list, layer_type, activation_fn):
    for layer in nn_list[:-1]:
        if layer[0] == layer_type:
            layer[2] = activation_fn

    return nn_list


def find_all_pairs_compatible_in_parent(parent, layer_prev, layer_next, is_first_layer):
    compatible_prev = []
    compatible_next = []

    for i, layer in enumerate(parent[:-1]):
        if is_first_layer and layer[0] == layer_prev[0]:
            compatible_prev.append(i)
        elif LayerType(layer[0]) in NN_STACKING_RULES[LayerType(layer_prev[0])]:
            compatible_prev.append(i)

        if layer_next[0] in NN_STACKING_RULES[LayerType(layer[0])]:
            compatible_next.append(i)

    return compatible_prev, compatible_next


def create_offspring_from_crossover_points(parent_1, parent_2, r1, r2, r3, r4, activation_fn=2):
    offspring = parent_1[:r1]
    offspring.extend(parent_2[r3:r4 + 1])
    offspring.extend(parent_1[r2 + 1:])
    offspring = copy.deepcopy(offspring)

    for layer in offspring:
        if NNArrayStructure.ACTIVATION_FUNCTION in ELEMENTS_FROM_ARRAY_USED_BY_LAYER[LayerType(layer[0])]:
            layer[2] = activation_fn

    return offspring


def layers_wise_distance_between_genomes(genome_1, genome_2):
    s1, s2 = sorted([genome_1, genome_2], key=len)
    distance = 0
    """
    for i in range(len(s2) - 1):
        len_layer = len(s1[i])
        layer_distance = [s1[i][j] - s2[i][j] for j in range(len_layer)]

        distance += np.linalg.norm(layer_distance, 2)
    """
    for s1_layer, s2_layer in list(zip(s1, s2))[:len(s2 - 1)]:
        layer_distance = np.subtract(s1_layer, s2_layer)

        distance += np.linalg.norm(layer_distance, 2)

    # each remaining layer in s1
    """
    for i in range(len(s2) - 1, len(s1) - 1):
        len_layer = len(s[i])
        layer_distance = [s[i][j] for j in range(len_layer)]

        distance += np.linalg.norm(layer_distance, 2)
    """
    for layer in s1[len(s2) - 1:len(s1 - 1)]:
        distance += np.linalg.norm(layer, 2)

    return distance


class NNOptimize:
    def __init__(self, problem_type=1, architecture_type=1, cross_validation_ratio=0.4,
                 mutation_probability=0.4, add_more_layers_prob=0.5, nn_size_scaler=0.5,
                 population_size=10, tournament_size=10, max_similar_models=3, training_epochs=10,
                 max_generations=10, total_experiments=5, max_layers=10):
        self.problem_type = problem_type if problem_type in (1, 2) else 1
        self.architecture_type = architecture_type if architecture_type in (1, 2) else 1
        self.cross_validation_ratio = cross_validation_ratio
        self.mutation_probability = mutation_probability
        self.add_more_layers_prob = add_more_layers_prob
        self.nn_size_scaler = nn_size_scaler
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.max_similar_models = max_similar_models
        self.training_epochs = training_epochs
        self.max_generations = max_generations
        self.total_experiments = total_experiments
        self.max_layers = max_layers

        self.population = []

    def getOutputLayer(self, output_shape):
        output_list = [0] * 8
        output_list[1] = output_shape[2]
        if self.problem_type == 1:
            activation_function = ActivationFunction.SIGMOID if output_shape[0] == 1 \
                else ActivationFunction.SOFTMAX
        else:
            activation_function = ActivationFunction.LINEAR
        output_list[2] = activation_function.value

        return output_list

    def generate_initial_population(self, input_shape, output_shape):
        output_list = self.getOutputLayer(output_shape)

        return [nn_genome.NNGenome(input_shape, output_list,
                                   self.architecture_type,
                                   self.add_more_layers_prob)
                for _ in range(self.population_size)]

    def get_population_fitness(self, population, X, y):
        scores = [build_model.partialy_train(genome.genome, X, y, self.training_epochs, self.cross_validation_ratio)
                  for genome in population]
        performance_score, weights = list(zip(*scores))

        map(rescale_size_of_nn, weights)
        scaled_performance_score = performance_score / np.linalg.norm(performance_score)

        fitness = 10 * (1 - self.nn_size_scaler) * scaled_performance_score + self.nn_size_scaler * weights

        return fitness, np.argmin(fitness)

    def tournament_selection(self, population_with_fitness):
        parents = []
        for ind in range(len(population_with_fitness) // 2):
            if population_with_fitness[ind * 2][1] < population_with_fitness[ind * 2 + 1][1]:
                new_parent = population_with_fitness[ind * 2][0]
            else:
                new_parent = population_with_fitness[ind * 2 + 1][0]
            parents.append(new_parent)

        return parents

    def selection(self, population, fitness_values):
        parents = []
        while len(population) < 2 * self.population_size:
            tournament_members = np.random.choice(zip(population, fitness_values), self.tournament_size)
            parents.append(self.tournament_selection(tournament_members))

        return parents

    def crossover(self, parent_1, parent_2):
        # we exclude last layer in crossover because it is the same in every nn in population
        len_parent_1 = len(parent_1.genome) - 1
        r1 = r2 = r3 = r4 = 0
        succes = False

        for _ in range(self.max_similar_models):
            r1, r2 = sorted(np.random.randint(len_parent_1, 2))
            if r1 == r2:
                r2 = len_parent_1 - 1

            is_first_layer = r1 == 0
            if is_first_layer:
                layer_prev = parent_1[r1]
            else:
                layer_prev = parent_1[r1 - 1]
            layer_next = parent_1[r1 - 1]

            compatible_prev, compatible_next = find_all_pairs_compatible_in_parent(parent_2, layer_prev, layer_next)

            compatible = []
            if not compatible_prev or not compatible_prev:
                compatible = [(i, j) for i in compatible_prev for j in compatible_next if 0 <= j - i < self.max_layers]

                if not compatible:
                    random_pair = np.random.choice(compatible)
                    r3, r4 = random_pair
                    succes = True
                    break

        return r1, r2, r3, r4, succes

    def crossover_population(self, parents):
        offsprings = []

        for i in range(len(parents) // 2):
            parent_1 = parents[2 * i]
            parent_2 = parents[2 * i + 1]

            r1, r2, r3, r4, success = self.crossover(parent_1.genome, parent_2.genome)

            if not success:
                continue

            activation_fn = parent_1.getActivationFunction()
            offspring = create_offspring_from_crossover_points(parent_1.genome, parent_2.genome,
                                                               r1, r2, r3, r4, activation_fn)

            offsprings.append(nn_genome.NNGenome(parent_1.input_shape, parent_1.output_layer,
                                                 self.architecture_type, self.add_more_layers_prob,
                                                 self.max_layers, genome=offspring))

        return offsprings

    def mutation(self, nn_genome, layer_ind):
        layer = nn_genome[layer_ind]
        next_layer = nn_genome[layer_ind + 1]

        if layer[0] == LayerType.POOLING.value:
            rnd_layer_elem_type = NNArrayStructure.POOLING_SIZE
        elif layer[0] == LayerType.DROPOUT.value:
            rnd_layer_elem_type = NNArrayStructure.DROPOUT_RATE
        else:
            # we need to add dropout layer in this case
            rnd_layer_elem_type = np.random.choice(
                set(ELEMENTS_FROM_ARRAY_USED_BY_LAYER[LayerType(layer[0])]) | {NNArrayStructure.DROPOUT_RATE})

        rnd_layer_elem = nn_genome.generate_layer_elem[rnd_layer_elem_type.value]()

        # we can't get invalid rnd_layer_elem
        if rnd_layer_elem_type != NNArrayStructure.DROPOUT_RATE:
            layer[rnd_layer_elem_type.value] = rnd_layer_elem
            if rnd_layer_elem_type == NNArrayStructure.ACTIVATION_FUNCTION:
                nn_genome = rectify_activation_function_nn(nn_genome, layer[0], rnd_layer_elem)
        elif layer[0] != LayerType.DROPOUT.value:
            if next_layer[0] == LayerType.DROPOUT.value:
                next_layer[NNArrayStructure.DROPOUT_RATE.value] = rnd_layer_elem
            else:
                nn_genome_tmp = nn_genome[:layer_ind + 1]
                new_dropout_layer = [LayerType.DROPOUT.value, 0, 0, 0, 0, 0, 0, rnd_layer_elem]
                nn_genome_tmp.append(new_dropout_layer)
                nn_genome_tmp.extend(nn_genome[layer_ind + 1:])
                nn_genome = copy.deepcopy(nn_genome_tmp)
        else:
            layer[NNArrayStructure.DROPOUT_RATE.value] = rnd_layer_elem

        return nn_genome

    def mutate_population(self, population):
        for genome in population:
            if np.random.random() > self.mutation_probability:
                rnd_index = np.random.randint(len(genome.genome) - 1)
                genome.genome = self.mutation(genome.genome, rnd_index)

    def is_generation_similar(self, population, similarity_treshold=0.9):
        len_population = len(population)
        distance_matrix = np.zeros(shape=(len_population, len_population))
        max_distance = 0
        max_pair = (0, 0)
        indices = ((i, j) for i in range(len_population) for j in range(len_population))

        for i, j in indices:
            dist = layers_wise_distance_between_genomes(population[i].genome, population[j].genome)
            distance_matrix[(i, j)] = dist

            if dist < max_distance:
                max_distance = dist
                max_pair = (i, j)

        if max_distance == 0:
            return True

        count_similar = 0
        normalize_fn = np.vectorize(lambda d: d / max_distance)
        distance_matrix = normalize_fn(distance_matrix)
        """
        for i, j in indices:
            if i != j and distance_matrix[(i, j)] < similarity_treshold and (i, j) != max_pair:
                count_similar += 1
        """
        count_similar = sum(1 for i, j in indices
                            if i != j and distance_matrix[(i, j)] < similarity_treshold and (i, j) != max_pair)

        if count_similar > self.max_similar_models:
            return True

        return False

    def find_best_model(self, X, y):
        best_models = []
        for _ in range(self.total_experiments):
            population = self.generate_initial_population(X.shape, y.shape)
            best_model = None
            best_fitness = float("inf")
            for _ in range(self.max_generations):
                population_fitness, best_model_index = self.get_population_fitness(population, X, y)

                if population_fitness[best_model_index] < best_fitness:
                    best_fitness = population_fitness[best_model_index]
                    best_model = population[best_model_index]

                parents = self.selection(population, population_fitness)
                population = self.crossover_population(parents)
                self.mutate_population(population)

                if self.is_generation_similar(population):
                    break

            best_models.append((best_models, best_fitness))

        return min(best_models, key=lambda model: model[1])

    def fit(self, X, y):
        best_model, best_fitness = self.find_best_model(X, y)
        print(f"Best model: {best_model}\nWith fitness = {best_fitness}")

        model = build_model.partialy_train(best_model.genome, X, y, self.training_epochs * 5,
                                           self.cross_validation_ratio, verbose=1, final_train=True)

        return model
