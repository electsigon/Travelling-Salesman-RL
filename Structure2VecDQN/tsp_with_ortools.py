from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

class Solver(object):

  def __init__(self, tsp_size):
    self.tsp_size = tsp_size          # The number of nodes
    self.num_routes = 1               # The number of routes, which is 1 for TSP
    self.depot = 0                    # The depot is the starting node of the route

  def run(self, dist_matrix):
    # Create routing model

    manager = pywrapcp.RoutingIndexManager(self.tsp_size, self.num_routes, self.depot)
    routing = pywrapcp.RoutingModel(manager)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    # Setting first solution heuristic: the
    # method for finding a first solution to the problem.
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    ################################################
    #                                              #
    #               TO EXTEND SEARCH               #
    #                                              #
    ################################################
    """
    # Setting guided local search in order to find the optimal solution
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit_ms = 40000
    """

    # Create the distance callback, which takes two arguments (the from and to node indices)
    # and returns the distance between these nodes.

    class DistanceCallback(object):
      def __init__(self, my_manager, matrix):
        self.matrix = matrix
        self.manager = my_manager

      def Distance(self, from_index, to_index):
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return int(10000 * self.matrix[from_node][to_node])

    dist_between_nodes = DistanceCallback(manager, dist_matrix)
    dist_callback = dist_between_nodes.Distance

    transit_callback_index = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Solve, returns a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)

    # Inspect solution. Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
    route_number = 0
    route_distance = 0
    index = routing.Start(route_number) # Index of the variable for the starting node.
    route = []
    plan_output = "Route: \n"
    while not routing.IsEnd(index):
      # Convert variable indices to node indices in the displayed route.
      plan_output += " {}->".format(manager.IndexToNode(index))
      route.append(manager.IndexToNode(index))
      previous_index = index
      index = assignment.Value(routing.NextVar(index))
      route_distance += dist_matrix[previous_index % self.tsp_size][index % self.tsp_size]
    route.append(manager.IndexToNode(index))
    plan_output += " {}\n".format(manager.IndexToNode(index))
    print(plan_output)

    return route, route_distance # route, optimal distance
