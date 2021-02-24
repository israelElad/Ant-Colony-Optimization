import random
from typing import Dict, List
import numpy as np
from Graph import Graph
# python 3.7+ required


def cummulative_total_cost(ant, new_node):
    new_total_cost = ant.get_total_cost() + \
        ant.graph.matrix[ant.get_current_node()][new_node]
    return new_total_cost


class AntColonyOptimizer():
    def __init__(
            self, num_of_ants: int, epochs: int, alpha: float, beta: float,
            evaporation_rate: float, Q: int, total_cost_func):
        """
        :param num_of_ants:
        :param epochs:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param evaporation_rate: pheromone residual coefficient
        :param q: pheromone intensity
        """
        self.num_of_ants = num_of_ants
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.total_cost_func = total_cost_func

    def solve(self, graph: Graph, generate_ants, verbose: bool = False):
        """
        :param graph:
        """
        best_global_cost = float('inf')
        best_global_path = []
        best_costs_per_epochs = []
        avg_costs_per_epochs = []

        plot_y = {"ACO- best cost": []}
        for epoch in range(self.epochs):
            ants = generate_ants(self, graph)
            curr_cost = []
            for ant in ants:
                start = ant.generate_start_node()
                ant.make_first_move(start)
                while not ant.has_completed_cycle():
                    next_node = ant.choose_next_node()
                    updated_cost = self.total_cost_func(ant, next_node)
                    ant.update_path_total_cost(updated_cost)
                    ant.move_to_node(next_node)
                updated_cost = self.total_cost_func(ant, start)
                ant.update_path_total_cost(updated_cost)
                curr_cost.append(ant.get_total_cost())
                if ant.get_total_cost() < best_global_cost:
                    best_global_cost = ant.get_total_cost()
                    best_global_path = ant.path()
                ant.update_pheromone_delta()
            self._update_pheromone(graph, ants)
            best_costs_per_epochs.append(best_global_cost)
            avg_costs_per_epochs.append(np.mean(curr_cost))
            if verbose:
                print('Generation #{} best cost: {}, path: {}'.format(
                    epoch+1, best_global_cost, best_global_path))
            plot_y["ACO- best cost"].append(best_global_cost)
        return best_global_cost, best_costs_per_epochs, avg_costs_per_epochs

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.evaporation_rate
                for ant in ants:
                    graph.pheromone[i][j] += ant.get_pheromone_delta_matrix()[i][j]


class Ant():
    def __init__(self, algorithm, graph):
        self.algorithm = algorithm
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta_matrix = []  # the local increase of pheromone
        # nodes which are allowed for the next selection
        self.allowed = list(range(graph.num_of_nodes))
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j]
                     for i in range(graph.num_of_nodes)]
                    for j in range(graph.num_of_nodes)]  # heuristic information

    def generate_start_node(self):
        return random.randint(0,
                              self.graph.num_of_nodes - 1)

    def get_total_cost(self):
        return self.total_cost

    def get_current_node(self):
        return self.current_node

    def make_first_move(self, start):
        self.current_node = start
        self.move_to_node(start)

    def path(self):
        return self.tabu

    def has_completed_cycle(self):
        return len(self.allowed) == 0

    def choose_next_node(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current_node][i] ** self.algorithm.alpha * self.eta[self.current_node][
                i] ** self.algorithm.beta
        # probabilities for moving to a node in the next step
        probabilities = [0 for _ in range(self.graph.num_of_nodes)]
        for i in range(self.graph.num_of_nodes):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current_node][i] ** self.algorithm.alpha * \
                    self.eta[self.current_node][i] ** self.algorithm.beta / \
                    denominator
            except ValueError:
                pass
        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        return selected

    def update_path_total_cost(self, cost):
        self.total_cost = cost

    def move_to_node(self, node):
        if node in self.allowed:
            self.allowed.remove(node)
        self.tabu.append(node)
        self.current_node = node

    def update_pheromone_delta(self):
        self.pheromone_delta_matrix = np.zeros(
            (self.graph.num_of_nodes, self.graph.num_of_nodes))
        for i in range(len(self.tabu)-1):
            visited_node = self.tabu[i]
            next_visited_node = self.tabu[i+1]
            self.pheromone_delta_matrix[visited_node][next_visited_node] = self.algorithm.Q / self.total_cost

    def get_pheromone_delta_matrix(self):
        return self.pheromone_delta_matrix

class Ant_Optional():
    def __init__(self, algorithm, graph, optional_nodes):
        self.internal_ant = Ant(algorithm, graph)
        self.graph = graph
        self.optional_nodes = optional_nodes
        self.mandatory_nodes = list(
            set(range(graph.num_of_nodes)) - set(optional_nodes))

        self.visited_all_mandatory = False
        self.completed_cycle = False

    def get_current_node(self):
        return self.internal_ant.get_current_node()

    def update_path_total_cost(self, cost):
        self.internal_ant.update_path_total_cost(cost)

    def update_pheromone(self, graph: Graph, ants: list):
        self.internal_ant.update_pheromone(graph, ants)

    def generate_start_node(self):
        return random.choice(self.mandatory_nodes)

    def make_first_move(self, start):
        self.start = start
        self.internal_ant.current_node = start
        self.internal_ant.move_to_node(start)
        self.ctr_visited_mandatory = 1

    def has_completed_cycle(self):
        return self.completed_cycle

    def choose_next_node(self):
        if not self.visited_all_mandatory and (self.ctr_visited_mandatory == len(self.mandatory_nodes)):
            self.visited_all_mandatory = True
            start_node = self.internal_ant.tabu.pop(0)
            self.internal_ant.allowed.append(start_node)  # TODO encapsulation!
        return self.internal_ant.choose_next_node()

    def move_to_node(self, node):
        self.internal_ant.move_to_node(node)
        if node not in self.optional_nodes:
            self.ctr_visited_mandatory += 1
        if self.visited_all_mandatory and node == self.start:
            self.completed_cycle = True

    def get_total_cost(self):
        return self.internal_ant.get_total_cost()

    def path(self):
        return [self.internal_ant.tabu[-1]]+self.internal_ant.tabu[:-1]

    def update_pheromone_delta(self):
        self.internal_ant.update_pheromone_delta()

    def get_pheromone_delta_matrix(self):
        return self.internal_ant.get_pheromone_delta_matrix()
