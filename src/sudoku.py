#
# SUDOKU EVOLUTIONARY ALGORITHM
#

from random import choice, random
import numpy as np
from timeit import default_timer as timer
import hashlib

### CONSTANTS ###

NINE = 9  # number of rows, cols and squares, etc. -> very important in sudoku
INDIVIDUAL_SIZE = NINE * NINE  # 9*9 = 81

numbers = range(1, NINE + 1)  # all numbers used in the sudoku

### PARAMETERS ###
DEAD_END_IND_TOLERANCE = 5
DEAD_END_FIT_TOLERANCE = 10

SELECTION_PROBABILITY = 0.6  # probability that best individual of tournament gets selected
TOURNAMENT_SIZE = 2  # number of individuals per tournament
BEST_FEW = 0.1  # percentage of best individuals of the population which gets selected anyway

NUMBER_GENERATION = 100000  # maximum generation which can be reached: possible termination criteria
MAX_TIME = 60 * 20  # maximum amount of seconds the algorithm may take: chosen termination criteria
POPULATION_SIZE = 500  # number of individuals per generation: optimized for 500
TRUNCATION_RATE = 0.5  # best 50% of the population is kept
MUTATION_RATE = 1.0 / INDIVIDUAL_SIZE  # probability to change a specific number during mutation
SUPER_MUTATION_RATE = 20.0 / INDIVIDUAL_SIZE  # probability to change a specific number during supermutation

### EVOLUTIONARY ALGORITHM ###

blacklist = []


# Biggest Difficulty: Premature divergence towards local optima, so no solution can be found.
# Solution approach: maintain big diversity and converge slowly so that hopefully most of the landscape can be explored.
# TODO maintain diversity to prevent premature divergence e.g. Multipopulation GA
def evolve(population_size, grid, presolving):
    sudoku = parse_input(grid)
    if presolving:
        sudoku = presolve(sudoku)
    # initial_sudoku = np.copy(sudoku)

    generation = 0
    dead_end_ind_counter = 0
    dead_end_fit_counter = 0
    last_best_ind = 0
    last_fit = 100

    population = create_pop(sudoku, population_size)
    fitness_population = evaluate_pop(population)

    # for gen in range(NUMBER_GENERATION):
    start = timer()
    while (timer() - start) < MAX_TIME:
        print("Time passed:", timer() - start)
        generation += 1
        mating_pool = select_pop(population, fitness_population, population_size, sudoku)
        offspring_population = crossover_pop(mating_pool, population_size)
        population = mutate_pop(offspring_population, sudoku, MUTATION_RATE)
        fitness_population = evaluate_pop(population)
        best_ind, best_fit = best_pop(population, fitness_population)
        print("\n#%5d" % generation, "fitness:%3d" % best_fit, "diversity:", unique_population_size(population), "/",
              population_size)
        print("".join(np.array_str(best_ind)))
        if best_fit == 0:
            print("Solution found.")
            return generation
        if best_fit < 6:
            if False:
                best_ind_hash = hash(best_ind)
                if best_ind_hash == last_best_ind:
                    dead_end_ind_counter += 1
                    if dead_end_ind_counter > DEAD_END_IND_TOLERANCE:
                        population = dead_end(best_ind, DEAD_END_IND_TOLERANCE, offspring_population, sudoku)
                        dead_end_ind_counter = 0
                else:
                    dead_end_ind_counter = 0

            if best_fit == last_fit:
                dead_end_fit_counter += 1
                if dead_end_fit_counter > DEAD_END_FIT_TOLERANCE:
                    population = dead_end(best_ind, DEAD_END_FIT_TOLERANCE, offspring_population, sudoku)
                    dead_end_fit_counter = 0
            else:
                dead_end_fit_counter = 0

            # last_best_ind = best_ind_hash
            last_fit = best_fit

    print("No solution found after", generation, "generations and ", MAX_TIME, "s")
    return generation


def dead_end(best_ind, tolerance, offspring_population, initial_sudoku):
    print("Probable dead end found after", tolerance, "generations with same best fitness.")
    # retrievable genetic algorithm: if dead end is found -> start over
    # print "Start over with new random population."
    # evolve()

    # blacklist: if dead end is found -> insert sudoku into blacklist
    add_to_blacklist(best_ind)
    print("Individual added to blacklist.")
    print("The blacklist now contains", len(blacklist), "items.")

    # supermutation: if deadend is found apply big mutation to whole population
    print("Supermutation applied with super mutation rate", SUPER_MUTATION_RATE)
    return mutate_pop(offspring_population, initial_sudoku, SUPER_MUTATION_RATE)


### POPULATION-LEVEL OPERATORS ###

def create_pop(sudoku, population_size):
    return [create_ind(sudoku) for _ in range(population_size)]


def evaluate_pop(population):
    assert population
    return [evaluate_ind(individual) for individual in population]


def select_pop_old(population, fitness_population, population_size):
    assert population
    sorted_population = sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])
    return [individual for individual, fitness in sorted_population[:int(population_size * TRUNCATION_RATE)]]


def unique_population_size(population):
    sum = 0
    hashes = []
    for individual in population:
        data_hash = hash(individual)
        if data_hash not in hashes:
            sum += 1
            hashes.append(data_hash)
    return sum


def hash(individual):
    return hashlib.sha1(individual).digest()


def remove_duplicates(list):
    unique = []
    hashes = []
    for individual, fitness in list:
        data_hash = hash(individual)
        if data_hash not in hashes:
            unique.append((individual, fitness))
            hashes.append(data_hash)
    return unique


def add_to_blacklist(individual):
    data_hash = hash(individual)
    if data_hash not in blacklist:
        blacklist.append(data_hash)


def remove_blacklist_items(list, hashes):
    population = []
    for individual, fitness in list:
        data_hash = hash(individual)
        if data_hash not in hashes:
            population.append((individual, fitness))
    return population


def is_in_blacklist(individual):
    return hash(individual) in blacklist


# use tournament selection to prevent premature divergence
def select_pop(population, fitness_population, population_size, sudoku):
    assert population

    # fill up population
    current_population_size = len(population)
    for _ in range(current_population_size, population_size):
        individual = create_ind(sudoku)
        population.append(individual)
        fitness_population.append(evaluate_ind(individual))

    assert len(population) == population_size

    sorted_population = sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])

    assert sorted_population

    # delete elements contained in blacklist
    sorted_population = remove_blacklist_items(sorted_population, blacklist)

    assert sorted_population

    # eliminate duplicates to preserve diversity
    sorted_population = remove_duplicates(sorted_population)

    assert sorted_population

    mating_pool = [individual for individual, fitness in
                   sorted_population[:int(population_size * BEST_FEW)]]  # include best few for sure
    while len(sorted_population) > int(population_size * TRUNCATION_RATE):
        tournament = [choice(range(0, len(sorted_population))) for _ in range(0, TOURNAMENT_SIZE)]
        for ind in tournament:
            if random() < SELECTION_PROBABILITY:
                mating_pool.append(sorted_population[ind][0])
                sorted_population.pop(ind)
                break

    assert mating_pool
    return mating_pool


def crossover_pop(population, population_size):
    assert population

    return [crossover_ind(choice(population), choice(population)) for _ in range(population_size)]


def mutate_pop(population, initial_sudoku, mutation_rate):
    assert population

    return [mutate_ind(individual, initial_sudoku, mutation_rate) for individual in population]


def best_pop(population, fitness_population):
    assert population

    return sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])[0]


### INDIVIDUAL-LEVEL OPERATORS: REPRESENTATION & PROBLEM SPECIFIC ###

def create_ind_old(sudoku):
    individual = np.copy(sudoku)
    for row in range(0, NINE):
        for col in range(0, NINE):
            if sudoku[row][col] == 0:  # if the number is not fixed
                individual[row][col] = choice(numbers)

    return individual


# use every number only once per row and then only swap numbers around in row
def create_ind(sudoku):
    individual = np.copy(sudoku)
    for row in range(0, NINE):
        availableNumbers = list(set(numbers) - set(sudoku[row]))  # delete the numbers already taken in the row
        for col in range(0, NINE):
            if availableNumbers and sudoku[row][col] == 0:
                individual[row][col] = choice(availableNumbers)
                availableNumbers.remove(individual[row][col])

    assert not 0 in individual
    return individual


# fitness function
def evaluate_ind(individual):
    # calculate number of duplicate numbers
    number_of_duplicates = 0
    # calculate each rows, cols and squares difference to 45
    # difference_to_45 = 0
    for row in individual:
        number_of_duplicates += NINE - len(set(row))
        # difference_to_45 += abs(45 - np.sum(row))
    for col in individual.transpose():  # transpose matrix to iterate over cols
        number_of_duplicates += NINE - len(set(col))
        # difference_to_45 += abs(45 - np.sum(col))
    for row in range(0, NINE, 3):
        for col in range(0, NINE, 3):
            square = individual[row:row + 3, col:col + 3]  # get every little square
            # compute difference between nine and the number of unique numbers
            number_of_duplicates += NINE - len(np.unique(square))
            # number_of_duplicates += NINE - np.sum([number for number in np.bincount(square) if number == 1])
            # difference_to_45 += abs(45 - np.sum(square))

    return number_of_duplicates  # + int(difference_to_45 / 3)


# crossover
def crossover_ind_old(mother, father):
    child = np.copy(mother)
    for row in range(0, NINE):
        for col in range(0, NINE):
            # single point
            # point = choice(range(1, NINE * NINE))
            # if row * NINE + col > point:
            #    child[row][col] = father[row][col]

            # double point
            # startPoint = choice(range(1, NINE * NINE / 2))
            # endPoint = choice(range(startPoint, NINE * NINE))
            # if startPoint < row * NINE + col < endPoint:
            #   child[row][col] = father[row][col]

            # multipoint
            if random() < 0.5:
                child[row][col] = father[row][col]

    # only allow children which are not in the blacklist
    if not is_in_blacklist(child):
        return child
    return choice([mother, father])


# crossover
# first randomly choose which type of crossover: row, col or square
# take mother and then by 50% chance take fathers row
def crossover_ind(mother, father):
    child = np.copy(mother)
    type = choice(['row', 'col', 'square'])
    # type = 'row'
    if type == 'row':
        for row in range(0, NINE):
            if random() < 0.5:
                child[row] = father[row]
    if type == 'col':
        for col in range(0, NINE):
            if random() < 0.5:
                child.transpose()[col] = father.transpose()[col]
    if type == 'square':
        for row in range(0, NINE, 3):
            for col in range(0, NINE, 3):
                if random() < 0.5:
                    child[row:row + 3, col:col + 3] = father[row:row + 3, col:col + 3]
    # only allow children which are not in the blacklist
    if not is_in_blacklist(child):
        return child
    return choice([mother, father])


# mutation
def mutate_ind_old(individual, initial_sudoku):
    for row in range(0, NINE):
        for col in range(0, NINE):
            if initial_sudoku[row][col] == 0:  # if the number is not fixed
                if random() < MUTATION_RATE:
                    # choose random number -> better
                    individual[row][col] = choice(numbers)

                    # choose number in range of original number
                    # if individual[row][col] == 9:
                    #     individual[row][col] = 8
                    # if individual[row][col] == 1:
                    #     individual[row][col] = 2
                    # else:
                    #     individual[row][col] = choice([individual[row][col]-1, individual[row][col]+1]) # choose number one below or above

    return individual


# mutation
# first randomly choose which type of mutation: row, col or square
# swap two numbers in row, col or square
def mutate_ind(individual, initial_sudoku, mutation_rate):
    if evaluate_ind(individual) < 6:
        mutation = mutate_swap_duplicates(individual, initial_sudoku)
    else:
        type = choice(['row', 'col', 'square'])
        # type = 'row'
        if type == 'row':
            mutation = mutate_row(individual, initial_sudoku, mutation_rate)

        if type == 'col':
            mutation = mutate_col(individual, initial_sudoku, mutation_rate)

        if type == 'square':
            mutation = mutate_square(individual, initial_sudoku, mutation_rate)
    # only allow mutations which are not in the blacklist
    if not is_in_blacklist(mutation):
        return mutation
    return individual


# first look for duplicates in rows and cols
# swap different numbers in other type
# so if duplicates are found in rows -> swap different numbers in col
def mutate_swap_duplicates(individual, initial_sudoku):
    number_set = set(numbers)
    swap_duplicates_in_row(individual, initial_sudoku, number_set)
    swap_duplicates_in_col(individual, initial_sudoku, number_set)
    return individual


def swap_duplicates_in_row(individual, initial_sudoku, number_set):
    # find duplicates
    coords = []  # stores the coordinates of the duplicates
    for row in range(0, NINE):
        row_set = set(individual[row])
        if len(row_set) < 9:  # there is at least one duplicate number here
            duplicate = list((number_set - row_set))[0]
            # get the coordinates of the duplicate numbers
            temp_indices = [(row, col) for col, number in enumerate(individual[row].tolist()) if number == duplicate]
            for row, col in temp_indices:
                if initial_sudoku[row][col]:  # if the duplicate number is not fixed
                    coords.append((row, col))
    # swap one duplicate pair
    cols = [coord[1] for coord in coords]
    while len(cols) > 0:
        col = choice(cols)
        if cols.count(col) >= 2:  # there is at least one other (hopefully different) duplicate in that col
            rows = [coord[0] for coord in coords if coord[1] == col]
            individual[rows[0]][col], individual[rows[1]][col] = individual[rows[1]][col], individual[rows[0]][
                col]  # swap
            break  # mutation done
        else:
            cols = [coord[1] for coord in coords if coord[1] != col]

    return individual


def swap_duplicates_in_col(individual, initial_sudoku, number_set):
    return swap_duplicates_in_row(individual.transpose(), initial_sudoku.transpose(), number_set).transpose()


def mutate_row(individual, initial_sudoku, mutation_rate):
    for row in range(0, NINE):
        for col in range(0, NINE):
            if initial_sudoku[row][col] == 0:  # if the number is not fixed
                if random() < mutation_rate:
                    # all numbers which are not fixed in this row
                    availableNumbers = list(set(individual[row]) - set(initial_sudoku[row]))
                    if individual[row][col] in availableNumbers:
                        availableNumbers.remove(individual[row][col])  # swap with a different number
                        if len(availableNumbers) > 0:
                            # swap number with random other one in row
                            chosen = choice(availableNumbers)  # choose one of the available numbers
                            colIndex = individual[row].tolist().index(chosen)  # get the index of the chosen number
                            temp = individual[row][col]  # save value
                            individual[row][col] = chosen  # put chosen value at mutated place
                            individual[row][colIndex] = temp  # put the saved value in the other numbers place
                            break  # do not mutate this row any further
    return individual


def mutate_col(individual, initial_sudoku, mutation_rate):
    # mutate row with transposed matrices and transpose back again at the end
    return mutate_row(individual.transpose(), initial_sudoku.transpose(), mutation_rate).transpose()


def mutate_square(individual, initial_sudoku, mutation_rate):
    for row in range(0, NINE, 3):
        for col in range(0, NINE, 3):
            if initial_sudoku[row][col] == 0:  # if the number is not fixed
                if random() < mutation_rate:
                    square = getSquare(individual, row, col)
                    initial = getSquare(initial_sudoku, row, col).flat
                    # all numbers which are not fixed in this square
                    availableNumbers = list(set(square.flat) - set(initial))
                    if individual[row][col] in availableNumbers:
                        availableNumbers.remove(
                            individual[row][col])  # swap with a different number
                        if len(availableNumbers) > 0:
                            # swap number with random other one in square
                            temp = individual[row][col]  # save value
                            chosen = choice(
                                availableNumbers)  # choose one of the available numbers
                            indices = np.argwhere(square == chosen)[
                                0]  # get the indices of the chosen number
                            individual[row][
                                col] = chosen  # put chosen value at mutated place
                            individual[indices[0]][
                                indices[
                                    1]] = temp  # put the saved value in the other numbers place
                            break  # do not mutate this square any further
    return individual


### PRESOLVING ###
# solve part of sudoku before
def presolve(sudoku):
    stillPossible = True  # continue as long as there is the possibility to fill in a number directly
    while stillPossible:
        stillPossible = False
        for row in range(0, NINE):
            for col in range(0, NINE):
                availableNumbers = list(set(numbers) - set(sudoku[row]))  # delete numbers which already occur in row
                availableNumbers = list(
                    set(availableNumbers) - set(sudoku.transpose()[col]))  # delete numbers which already occur in col
                square = getSquare(sudoku, row, col).flat
                availableNumbers = list(
                    set(availableNumbers) - set(square))  # delete numbers which already occur in square
                if len(availableNumbers) == 1 and sudoku[row][col] == 0:
                    sudoku[row][col] = availableNumbers[0]
                    stillPossible = True
    return sudoku


def getSquare(sudoku, row, col):
    squareRow = row - (row % 3)
    squareCol = col - (col % 3)
    return sudoku[squareRow:squareRow + 3, squareCol:squareCol + 3]


### PARSING ###

def parse_input(filename):
    sudoku = []
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        if '-' not in line:
            delete = line.maketrans(dict.fromkeys("!"))
            line = line.translate(delete)
            line = line.replace(".", "0")
            line = line.strip()
            row = [int(i) for i in list(line)]
            sudoku.append(row)
    return np.array(sudoku)


### SETUP ###

# grids have to be in the same directory as this file
grid1 = 'grid1.ss.txt'
grid2 = 'grid2.ss.txt'
grid3 = 'grid3.ss.txt'


### EVOLVE! ###

# evolve(POPULATION_SIZE, grid2, True)  # if presolving should be turned of, just put False in here


### EXPERIMENTS! ###

def experiment_for_population_size(size, grid):
    solved_number = 0
    generations = 0
    time = 0
    ROUNDS = 5
    for x in range(1, ROUNDS + 1):
        start = timer()
        currentGenerations = evolve(size, grid, True)  # if presolving should be turned of, just put False in here
        generations += currentGenerations
        currentTime = timer() - start
        if currentTime < MAX_TIME:
            solved_number += 1
        time += currentTime
        print(str(x) + "th Run: Number of Generations: " + str(currentGenerations) + ", " + str(
            round(currentTime, 3)) + "s")
    summary = "Average Number of Generations: " + str(generations / ROUNDS) + ", Algorithm running for: " + str(
        round(time, 3)) + "s, Number of solved Sudokus: " + str(solved_number) + "/" + str(ROUNDS) + "\n"
    print(summary)
    return summary


def experiment_for_grids(size):
    text = "========== Population Size: " + str(size) + " =========="
    print(text)
    summary = text + "\n"

    text = "===== Grid 3 ====="
    print(text)
    summary += text + "\n"
    summary += experiment_for_population_size(size, grid3) + "\n"

    text = "===== Grid 2 ====="
    print(text)
    summary += text + "\n"
    summary += experiment_for_population_size(size, grid2) + "\n"

    text = "===== Grid 1 ====="
    print(text)
    summary += text + "\n"
    summary += experiment_for_population_size(size, grid1) + "\n"

    return summary


def experiment():
    print("==================== Start of Experiment ====================")
    summary = experiment_for_grids(10) + "\n"
    summary += experiment_for_grids(100) + "\n"
    summary += experiment_for_grids(1000) + "\n"
    summary += experiment_for_grids(10000) + "\n"
    print("==================== Experiment Summary ====================")
    print(summary)
    print("==================== End of Experiment ====================")


### EXPERIMENT! ###

experiment()

### DEBUGGING ###

if False:
    mother = create_ind()
    father = create_ind()

    print("create")
    print(mother)

    print("fitness")
    print(evaluate_ind(mother))

    child = crossover_ind(mother, father)
    print("crossover")
    print(child)

    print("mutate")
    print(mutate_ind(child))
