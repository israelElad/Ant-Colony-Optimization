
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

matrix = [[0, 51, 14, 48, 1], #best path: 0->2->3->1->0 = 100
          [28, 0, 19, 49, 100],
          [69, 92, 0, 30, 100],
          [85, 28, 100, 0, 100],
          [100, 100, 1, 100, 0]]

g = aco_optional.Graph(matrix, [4])

alg = aco_optional.ACO_Optional(100, 100, 1, 1, 0.5, 1, 0)

best_solution, best_cost, avg_costs, best_costs = alg.solve(g, True)

print(best_cost)
