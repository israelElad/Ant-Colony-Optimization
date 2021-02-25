class Graph():
    def __init__(self, cost_matrix):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        self.matrix = cost_matrix
        self.num_of_nodes = len(cost_matrix)
        self.pheromone = [[1 / (self.num_of_nodes * self.num_of_nodes)
                           for j in range(self.num_of_nodes)]
                          for i in range(self.num_of_nodes)]
