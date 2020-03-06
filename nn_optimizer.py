import numpy as np
import nn_genome
import build_model
from building_rules import LayerType, ActivationFunction


def replace_last_three_digits_with_zeros(number):
    round_to = 10 ** 3
    if number <= round_to:
        return number

    return round(number / round_to) * round_to


def rescale_size_of_nn(size):
    scaled_size = replace_last_three_digits_with_zeros(size)
    return np.log10(scaled_size)


class NNOptimize:
    def __init__(self, problem_type=1, architecture_type=1, cross_validation_ratio=0.4,
                 mutation_probability=0.4, add_more_layers_prob=0.5, nn_size_scaler=0.5,
                 population_size=10, tournament_size=10, max_similar_models=3, training_epochs=10,
                 max_generations=10, total_experiments=5):
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

        return 10 * (1 - self.nn_size_scaler) * scaled_performance_score + self.nn_size_scaler * weights

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
        for _ in range(self.max_similar_models):
            r1, r2 = sorted(np.random.randint(len_parent_1, 2))
            if r1 == r2:
                r2 = len_parent_1 - 1

    def mutate_population(self, population):
        for genome in population:
            if np.random.random() > self.mutation_probability:
                rnd_index = np.random.randint(len(genome.genome) - 1)
                genome.genome = self.mutation(genome.genome, rnd_index)

    def mutation(self, layer_ind):

        return self.mutation_probability

    def generate_offsprings(self):
        pass

    def fit(self, X, y):
        population = self.generate_initial_population(X.shape, y.shape)
