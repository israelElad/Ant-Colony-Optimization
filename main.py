import aco
import aco_max
import aco_combined
import aco_optional
from plot import Plot
import numpy as np
from matrices import matrices, all_num_optional_nodes

matrices_size=["Small Graph","Medium Graph", "Large Graph", "XL Graph"]
i=0
for matrix, num_optional_nodes in zip(matrices, all_num_optional_nodes):

    whole_matrix_rank = len(matrix)
    classical_matrix_rank = whole_matrix_rank - num_optional_nodes
    m = np.array(matrix)
    classical_matrix = m[0:classical_matrix_rank, 0:classical_matrix_rank]
    optional_nodes = list(range(classical_matrix_rank, whole_matrix_rank))

    g = aco.Graph(classical_matrix)
    alg = aco.ACO(10, 20, 1, 1, 0.5, 1, 0)
    best_solution, best_cost, avg_costs, best_costs, plot_x, plot_y = alg.solve(g, True)

    # new_plot=Plot("Regular ACO" ,['r', 'g'],plot_x,"cost")
    # new_plot.plot_lines(plot_y)
    # new_plot.display()

    g0 = aco_max.Graph(classical_matrix)
    alg0 = aco_max.ACO_Max(10, 20, 1, 1, 0.5, 1, 0)
    best_solution0, best_cost0, avg_costs0, best_costs0, plot_x0, plot_y0= alg0.solve(g0, True)

    # new_plot=Plot("Max ACO",['r', 'g'],plot_x0,"cost")
    # new_plot.plot_lines(plot_y0)
    # new_plot.display()

    g1 = aco_optional.Graph(matrix, optional_nodes)
    alg1 = aco_optional.ACO_Optional(10, 20, 1, 1, 0.5, 1, 0)
    best_solution1, best_cost1, avg_costs1, best_costs1, plot_x1, plot_y1 = alg1.solve(g1, True)

    # new_plot=Plot("Optional ACO",['r', 'g'],plot_x1,"cost")
    # new_plot.plot_lines(plot_y1)
    # new_plot.display()

    g2 = aco_combined.Graph(matrix, optional_nodes)
    alg2 = aco_combined.ACO_Combined(10, 20, 1, 1, 0.5, 1, 0)
    best_solution2, best_cost2, avg_costs2, best_costs2, plot_x2, plot_y2 = alg2.solve(g2, True)

    # new_plot=Plot("Combined ACO",['r', 'g'],plot_x2,"cost")
    # new_plot.plot_lines(plot_y2)
    # new_plot.display()

    new_plot=Plot(matrices_size[i] ,['r', 'g','b','yellow'],plot_x,"cost")
    new_plot.plot_lines({**plot_y, **plot_y0,**plot_y1,**plot_y2})
    new_plot.display()

    i+=1



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
