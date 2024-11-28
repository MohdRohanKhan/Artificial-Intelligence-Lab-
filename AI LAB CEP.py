import random

# Genetic Algorithm Parameters
POP_SIZE = 6  # Population size
CHROMOSOME_LENGTH = 5  # Number of bits in chromosome
GENERATIONS = 10  # Number of generations
MUTATION_RATE = 0.1  # Probability of mutation


# Fitness function: f(x) = x^2
def fitness_function(x):
    return x ** 2


# Decode binary chromosome to integer
def decode_chromosome(chromosome):
    return int(''.join(map(str, chromosome)), 2)


# Generate initial population
def generate_population():
    return [[random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)] for _ in range(POP_SIZE)]


# Roulette wheel selection
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, fitness in enumerate(fitnesses):
        current += fitness
        if current > pick:
            return population[i]


# Crossover operation
def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


# Mutation operation
def mutate(chromosome):
    return [
        bit if random.random() > MUTATION_RATE else 1 - bit for bit in chromosome
    ]


# Genetic Algorithm
def genetic_algorithm():
    population = generate_population()
    for generation in range(GENERATIONS):
        # Calculate fitness for the population
        fitnesses = [fitness_function(decode_chromosome(individual)) for individual in population]

        # Print best solution of the generation
        best_individual = max(population, key=lambda ind: fitness_function(decode_chromosome(ind)))
        best_fitness = fitness_function(decode_chromosome(best_individual))
        print(
            f"Generation {generation}: Best Fitness = {best_fitness}, Best Individual = {decode_chromosome(best_individual)}")

        # Generate new population
        new_population = []
        for _ in range(POP_SIZE // 2):
            # Select parents
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            # Perform crossover
            offspring1, offspring2 = crossover(parent1, parent2)
            # Perform mutation
            new_population.extend([mutate(offspring1), mutate(offspring2)])
        population = new_population


# Run the Genetic Algorithm
genetic_algorithm()
