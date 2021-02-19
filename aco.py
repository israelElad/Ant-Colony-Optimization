import random
from typing import Dict, List
import numpy as np
from Graph import Graph
# python 3.7+ required


def cummulative_total_cost(ant, new_node):
    new_total_cost = ant.total_cost + \
        ant.graph.matrix[ant.current_node][new_node]
    return new_total_cost


class AntColonyOptimizer():
    def __init__(
            self, num_of_ants: int, epochs: int, alpha: float,
            beta: float, evaporation_rate: float, Q: int, total_cost_func):
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


    def solve(self, graph: Graph, verbose: bool = False):
        """
        :param graph:
        """
        best_global_cost = float('inf')
        best_global_path = []
        best_costs_per_epochs = []
        avg_costs_per_epochs = []

        plot_y = {"ACO- best cost": []}
        for epoch in range(self.epochs):
            ants = [_Ant(self, graph) for _ in range(self.num_of_ants)]
            curr_cost = []
            for ant in ants:
                for _ in range(graph.num_of_nodes - 1):
                    next_node = ant.choose_next_node()
                    ant.move_to_node(next_node)
                ant.return_to_start()
                curr_cost.append(ant.total_cost)
                if ant.total_cost < best_global_cost:
                    best_global_cost = ant.total_cost
                    best_global_path = [] + ant.tabu
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
                    graph.pheromone[i][j] += ant.pheromone_delta_matrix[i][j]


class _Ant():
    def __init__(self, algorithm, graph):
        self.algorithm = algorithm
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta_matrix = []  # the local increase of pheromone
        # nodes which are allowed for the next selection
        self.allowed = [i for i in range(graph.num_of_nodes)]
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j]
                     for j in range(graph.num_of_nodes)]
                    for i in range(graph.num_of_nodes)]  # heuristic information
        start = random.randint(0,
                               graph.num_of_nodes - 1)  # start from any node
        # print(start)
        self.tabu.append(start)
        self.current_node = start
        self.allowed.remove(start)

    def choose_next_node(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current_node][i] ** self.algorithm.alpha * self.eta[self.current_node][
                i] ** self.algorithm.beta
        # probabilities for moving to a node in the next step
        probabilities = [0 for i in range(self.graph.num_of_nodes)]
        for i in range(self.graph.num_of_nodes):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current_node][i] ** self.algorithm.alpha * \
                    self.eta[self.current_node][i] ** self.algorithm.beta / \
                    denominator
            except ValueError:
                pass  # do nothing
        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        return selected

    def move_to_node(self, node):
        self.allowed.remove(node)
        self.tabu.append(node)
        self.total_cost = self.algorithm.total_cost_func(self, node)
        self.current_node = node

    def return_to_start(self):
        start_node = self.tabu[0]
        self.total_cost = self.algorithm.total_cost_func(self, start_node)

    def update_pheromone_delta(self):
        self.pheromone_delta_matrix = np.zeros(
            (self.graph.num_of_nodes, self.graph.num_of_nodes))
        for i in range(len(self.tabu)-1):
            visited_node = self.tabu[i]
            next_visited_node = self.tabu[i+1]
            self.pheromone_delta_matrix[visited_node][next_visited_node] = self.algorithm.Q / self.total_cost
