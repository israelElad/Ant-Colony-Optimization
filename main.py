
import random

import numpy
import aco_optional
from aco import ACO, Graph


# matrix =[[0, 51, 14, 48],
#          [28, 0, 19, 49],
#          [69, 92, 0, 30],
#          [85, 28, 100, 0]]

# g = Graph(matrix, 4)

# alg = ACO(2, 10, 1, 1, 0.5, 1, 0)

# best_solution, best_cost, avg_costs, best_costs = alg.solve(g, True)

# print(best_cost)

# matrix = [[0, 51, 14, 48, 1], #best path: 0->2->3->1->0 = 100
#           [28, 0, 19, 49, 100],
#           [69, 92, 0, 30, 100],
#           [85, 28, 100, 0, 100],
#           [100, 100, 1, 100, 0]]

matrix = [[0, 8, 47, 73, 89, 61, 89, 64, 59, 43], [94, 0, 63, 68, 96, 7, 44, 40, 27, 75], [17, 6, 0, 50, 65, 83, 81, 81, 45, 76], [41, 68, 91, 0, 71, 53, 33, 98, 17, 37], [23, 46, 35, 54, 0, 77, 30, 60, 40, 65], [
    39, 40, 76, 1, 5, 0, 84, 1, 70, 20], [70, 83, 58, 47, 45, 28, 0, 88, 13, 34], [34, 57, 88, 72, 70, 70, 96, 0, 65, 23], [52, 67, 47, 35, 69, 58, 87, 27, 0, 77], [11, 51, 99, 36, 41, 64, 12, 47, 37, 0]]

print(numpy.matrix(matrix))


# for i in range(10):
#     matrix.append([])
#     for j in range(10):
#         if i==j:
#             matrix[i].append(0)
#         else:
#             matrix[i].append(random.randint(1, 100))

# print(matrix)

g = aco_optional.Graph(matrix, [4])

alg = aco_optional.ACO_Optional(10, 1000, 1, 1, 0.5, 1, 0)

best_solution, best_cost, avg_costs, best_costs = alg.solve(g, True)

print(best_cost)
