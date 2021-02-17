
import random
import numpy

import aco
import aco_max
import aco_combined
import aco_optional
from aco_max import ACO_Max, Graph


# matrix =[[0, 51, 14, 48],
#          [28, 0, 19, 49],
#          [69, 92, 0, 30],
#          [85, 28, 100, 0]]

# g = Graph(matrix, 4)

# alg = ACO(2, 10, 1, 1, 0.5, 1, 0)

# best_solution, best_cost, avg_costs, best_costs = alg.solve(g, True)

# print(best_cost)
from plot import Plot

from matrices import matrices, num_optional_nodes
import numpy as np
from aco import * 

for matrix, optional in zip(matrices, num_optional_nodes):
    optiona_matrix_rank = len(matrix) - optional
    m = np.array(matrix)
    optional_matrix = m[0:optiona_matrix_rank, 0:optiona_matrix_rank]
    g = aco.Graph(matrix)
    alg = AntColonyOptimizer(
        num_of_ants=10, epochs=10, total_cost_func=cummulative_total_cost,
        alpha=1, beta=1, evaporation_rate=0.5, q=1)
    best_solution, best_cost, best_costs, plot_x, plot_y = alg.solve(
        g, True)

    new_plot = Plot("Regular ACO", ['r', 'g', 'b'], plot_x, "cost")
    new_plot.plot_lines(plot_y)
    new_plot.display()

    g0 = aco_max.Graph(matrix)
    alg0 = aco_max.ACO_Max(10, 10, 1, 1, 0.5, 1, 0)
    best_solution0, best_cost0, avg_costs0, best_costs0, plot_data0 = alg0.solve(
        g0, True)

    new_plot = Plot("Max ACO")
    new_plot.plot_points(plot_data0, ['r', 'g', 'b'])
    new_plot.display()

    g1 = aco_optional.Graph(matrix, optional)
    alg1 = aco_optional.ACO_Optional(10, 10, 1, 1, 0.5, 1, 0)
    best_solution1, best_cost1, avg_costs1, best_costs1, plot_data1 = alg1.solve(
        g1, True)

    new_plot = Plot("Optional ACO")
    new_plot.plot_points(plot_data1, ['r', 'g', 'b'])
    new_plot.display()

    g2 = aco_combined.Graph(matrix, optional)
    alg2 = aco_combined.ACO_Combined(10, 10, 1, 1, 0.5, 1, 0)
    best_solution2, best_cost2, avg_costs2, best_costs2, plot_data2 = alg2.solve(
        g2, True)

    new_plot = Plot("Combined ACO")
    new_plot.plot_points(plot_data2, ['r', 'g', 'b'])
    new_plot.display()


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
