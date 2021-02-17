import random
from typing import Dict, List
import numpy as np


class Graph(object):
    def __init__(self, cost_matrix: list, optional_nodes):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        self.matrix = cost_matrix
        self.optional_nodes = optional_nodes
        self.rank = len(cost_matrix)
        self.mandatory_nodes = list(
            set([i for i in range(self.rank)]) - set(optional_nodes))
        self.pheromone = [[1 / (self.rank * self.rank)
                           for j in range(self.rank)] for i in range(self.rank)]


class ACO_Combined(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int):
        """
        :param ant_count:
        :param generations:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
        """
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    # noinspection PyProtectedMember
    def solve(self, graph: Graph, verbose: bool = False):
        """
        :param graph:
        """
        best_cost = float('inf')
        best_solution = []
        avg_costs = []
        best_costs = []
        plot_data = {"gen": [], "ACO Combined- average cost": [],
                     "ACO Combined- best cost": []}  # type: Dict[str, List[float]]
        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ant in ants:
                curr_cost = []
                while ant._has_unvisited_mandatory_nodes():
                    ant._select_next()
               # ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]] # todo
                ant.total_cost = max(ant.total_cost, graph.matrix[ant.tabu[-1]][ant.tabu[0]])
                curr_cost.append(ant.total_cost)
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + [ant.tabu[-1]]+ant.tabu[:-1]
                # update pheromone
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
            best_costs.append(best_cost)
            avg_costs.append(np.mean(curr_cost))
            if verbose:
                print('Generation #{} best cost: {}, avg cost: {}, path: {}'.format(
                    gen+1, best_cost, avg_costs[-1], best_solution))
            plot_data["gen"].append(gen+1)
            plot_data["ACO Combined- average cost"].append(avg_costs[-1])
            plot_data["ACO Combined- best cost"].append(best_cost)
        return best_solution, best_cost, avg_costs, best_costs, plot_data


class _Ant(object):
    def __init__(self, aco, graph):
        self.visited_all_mandatory = False
        self.completed_cycle = False
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []
        self.pheromone_delta = []  # the local increase of pheromone
        # nodes which are allowed for the next selection
        self.allowed = [i for i in range(graph.rank)]
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # heuristic information
        # start from any node
        index = random.randint(0, len(graph.mandatory_nodes) - 1)
        start = graph.mandatory_nodes[index]
        self.start = start
        self.ctr_visited_mandatory = 1        
        # print(start)
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _has_unvisited_mandatory_nodes(self):
        # print("mandatory ctr ", self.ctr_visited_mandatory)
        # print("other ", self.graph.rank-len(self.graph.optional_nodes))
        return not self.completed_cycle

    def _select_next(self):
        if not self.visited_all_mandatory and (self.ctr_visited_mandatory == len(self.graph.mandatory_nodes)): #todo - was +1
            self.allowed.append(self.tabu.pop(0))
            self.visited_all_mandatory = True
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                i] ** self.colony.beta
        # noinspection PyUnusedLocal
        # probabilities for moving to a node in the next step
        probabilities = [0 for i in range(self.graph.rank)]
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
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
        if selected not in self.graph.optional_nodes:
            self.ctr_visited_mandatory += 1
        self.tabu.append(selected)
        # print("allowed: ", self.allowed)
        # print("selected: ", selected)
        # print("     \n")
        self.allowed.remove(selected)
        self.total_cost = max(self.total_cost, self.graph.matrix[self.current][selected])
        self.current = selected
        if self.visited_all_mandatory and self.start == selected:
            self.completed_cycle = True

    # noinspection PyUnusedLocal
    def _update_pheromone_delta(self):
        self.pheromone_delta = [
            [0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                # noinspection PyTypeChecker
                self.pheromone_delta[i][j] = self.colony.Q / \
                    self.graph.matrix[i][j]
            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost
