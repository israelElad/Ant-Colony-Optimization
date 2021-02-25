
import random
import numpy

import aco
import aco_max
import aco_combined
import aco_optional


# matrix =[[0, 51, 14, 48],
#          [28, 0, 19, 49],
#          [69, 92, 0, 30],
#          [85, 28, 100, 0]]

# g = Graph(matrix, 4)

# alg = ACO(2, 10, 1, 1, 0.5, 1, 0)

# best_solution, best_cost, avg_costs, best_costs = alg.solve(g, True)

# print(best_cost)
from plot import Plot

from matrices import matrices, all_num_optional_nodes
import numpy as np
from aco import *
from Graph import Graph

num_of_ants = 100
epochs = 100
total_cost_func = cummulative_total_cost
alpha = 1
beta = 1
evaporation_rate = 0.5
Q = 1

def plot(best_costs_per_epochs, avg_costs_per_epochs, name):
    plot_x = {"epoch": range(1, epochs+1)}
    plot_y = {}
    plot_y["best cost"] = best_costs_per_epochs
    plot_y["average cost"] = avg_costs_per_epochs
    new_plot = Plot(name, ['r', 'g', 'b'], plot_x, "cost")
    new_plot.plot_lines(plot_y)
    new_plot.display()


def create_classical_ants_generator(num_of_ants):
    
    def f(algo, graph):    
        return [Ant(algo, graph) for _ in range(num_of_ants)]
    return f

def create_optional_ants_generator(num_of_ants, optional_nodes):
    def f(algo, graph):
        return [Ant_Optional(algo, graph, optional_nodes) for _ in range(num_of_ants)]
    return f

for matrix, num_optional_nodes in zip(matrices, all_num_optional_nodes):
    whole_matrix_rank = len(matrix)
    classical_matrix_rank = whole_matrix_rank - num_optional_nodes
    m = np.array(matrix)
    classical_matrix = m[0:classical_matrix_rank, 0:classical_matrix_rank]
    graph_for_classical = Graph(classical_matrix)
    graph_for_optional = Graph(matrix)
    optional_nodes = list(
        range(classical_matrix_rank, whole_matrix_rank)) #TODO
    
    alg = AntColonyOptimizer(
        num_of_ants, epochs,
        alpha, beta, evaporation_rate, Q,
        total_cost_func=cummulative_total_cost)
    best_global_cost, best_costs_per_epochs, avg_costs_per_epochs = alg.solve(
        graph=graph_for_classical, generate_ants=create_classical_ants_generator(num_of_ants), verbose=True)

    plot(best_costs_per_epochs, avg_costs_per_epochs, "classical ACO")
    

    alg = AntColonyOptimizer(
        num_of_ants, epochs,
        alpha, beta, evaporation_rate, Q,
        total_cost_func=cummulative_total_cost)
    best_global_cost, best_costs_per_epochs, avg_costs_per_epochs = alg.solve(
        graph=graph_for_optional, generate_ants=create_optional_ants_generator(num_of_ants, optional_nodes), verbose=True)

    plot(best_costs_per_epochs, avg_costs_per_epochs, "ACO with optional nodes")

    alg = AntColonyOptimizer(
        num_of_ants, epochs,
        alpha, beta, evaporation_rate, Q,
        total_cost_func=max_total_cost)
    best_global_cost, best_costs_per_epochs, avg_costs_per_epochs = alg.solve(
        graph=graph_for_classical, generate_ants=create_classical_ants_generator(num_of_ants), verbose=True)

    plot(best_costs_per_epochs, avg_costs_per_epochs, "ACO with max function")

    alg = AntColonyOptimizer(
        num_of_ants, epochs,
        alpha, beta, evaporation_rate, Q,
        total_cost_func=max_total_cost)
    best_global_cost, best_costs_per_epochs, avg_costs_per_epochs = alg.solve(
        graph=graph_for_optional, generate_ants=create_optional_ants_generator(num_of_ants, optional_nodes), verbose=True)

    plot(best_costs_per_epochs, avg_costs_per_epochs, "ACO with optional nodes with max function")
    # g0 = aco_max.Graph(matrix)
    # alg0 = aco_max.ACO_Max(10, 10, 1, 1, 0.5, 1, 0)
    # best_solution0, best_cost0, avg_costs0, best_costs0, plot_data0 = alg0.solve(
    #     g0, True)

    # new_plot = Plot("Max ACO")
    # new_plot.plot_points(plot_data0, ['r', 'g', 'b'])
    # new_plot.display()

    # g1 = aco_optional.Graph(matrix, optional)
    # alg1 = aco_optional.ACO_Optional(10, 10, 1, 1, 0.5, 1, 0)
    # best_solution1, best_cost1, avg_costs1, best_costs1, plot_data1 = alg1.solve(
    #     g1, True)

    # new_plot = Plot("Optional ACO")
    # new_plot.plot_points(plot_data1, ['r', 'g', 'b'])
    # new_plot.display()

    # g2 = aco_combined.Graph(matrix, optional)
    # alg2 = aco_combined.ACO_Combined(10, 10, 1, 1, 0.5, 1, 0)
    # best_solution2, best_cost2, avg_costs2, best_costs2, plot_data2 = alg2.solve(
    #     g2, True)

    # new_plot = Plot("Combined ACO")
    # new_plot.plot_points(plot_data2, ['r', 'g', 'b'])
    # new_plot.display()


# matrix=[]
# for i in range(50):
#     matrix.append([])
#     for j in range(50):
#         if i==j:
#             matrix[i].append(0)
#         else:
#             matrix[i].append(random.randint(1, 100))

# print(matrix)

# g = aco_optional.Graph(matrix, [4])

# alg = aco_optional.ACO_Optional(10, 1000, 1, 1, 0.5, 1, 0)

# best_solution, best_cost, avg_costs, best_costs = alg.solve(g, True)

# print(best_cost)


