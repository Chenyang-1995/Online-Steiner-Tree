import time
import numpy as np
import math

from collections import defaultdict

import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph
#from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, shortest_path
LARGE_INT = 100000
class vertex:
    def __init__(self,id):
        self.id = id
        self.neighbour = []
        self.edge_cost = {}

    def add_neighbour(self,id,cost,NeedCheck=False):

        if NeedCheck == False:
            self.neighbour.append(id)
            self.edge_cost[id] = cost
        else:
            if id not in self.neighbour:
                self.neighbour.append(id)
            self.edge_cost[id] = cost


class graph:
    def __init__(self):
        self.num_vertices = -1
        self.num_edges = -1
        self.vertex_set = []
        self.edge_set = []
        self.edge_cost = {}

    def load_data(self,FILE_NAME='road_graph/graph_0.txt'):
        df = pd.read_csv(FILE_NAME,sep=' ')
        self.data = df

        u_list = df['u'].unique()
        max_u = max(u_list)
        print(min(u_list))

        v_list = df['v'].unique()
        max_v = max(v_list)
        print(min(v_list))

        max_vertex = max([max_u,max_v])
        #print(max_vertex)
        self.num_vertices = max_vertex + 1
        self.num_edges = df.shape[0]
        print('Vertex = {}'.format(self.num_vertices))
        print('Edge = {}'.format(self.num_edges))
        self.orginal_graph = np.zeros(shape=(self.num_vertices,self.num_vertices),dtype= np.int_)

        for edge_index in range(self.num_edges):
            tmp_edge = df.iloc[edge_index]
            u = tmp_edge['u']
            v = tmp_edge['v']
            w = tmp_edge['w']
            self.orginal_graph[u][v] = w
            self.orginal_graph[v][u] = w

        self.csr_graph = csr_matrix(self.orginal_graph)

        self.shortest_distance = csgraph.shortest_path(self.csr_graph,directed=False)
        print(self.shortest_distance)

        self.radius = np.max(np.array(self.shortest_distance)) * 0.5


    def random_graph(self,num_vertices,num_edges):

        #print(max_vertex)
        self.num_vertices = num_vertices
        self.num_edges = num_edges
        print('Vertex = {}'.format(self.num_vertices))
        print('Edge = {}'.format(self.num_edges))

        potential_edges = [(x,y) for x in range(self.num_vertices-1) for y in range(x+1,self.num_vertices)]
        sampled_edge_induice = np.random.choice(len(potential_edges),size=self.num_edges,replace=False).tolist()
        sampled_edges = [ potential_edges[x] for x in sampled_edge_induice ]

        self.orginal_graph = np.ones(shape=(self.num_vertices,self.num_vertices),dtype= np.int_) * LARGE_INT

        for u,v in sampled_edges:

            w = np.random.randint(1,1001)
            self.orginal_graph[u][v] = w
            self.orginal_graph[v][u] = w

        self.csr_graph = csr_matrix(self.orginal_graph)

        self.shortest_distance = csgraph.shortest_path(self.csr_graph,directed=False)
        print(self.shortest_distance)

        self.radius = np.max(np.array(self.shortest_distance)) * 0.5


    def closest_distance(self,target_node,center_nodes):
        distance = self.shortest_distance[target_node][center_nodes[0]]
        node = center_nodes[0]
        for center_node in center_nodes:
            if self.shortest_distance[target_node][center_node] < distance:
                distance = self.shortest_distance[target_node][center_node]
                node = center_node

        return  distance, node
    def ratio_threhold_center_cluster(self,distance_ratio = 0.1):
        #self.threhold = threhold
        self.center_nodes = []


        self.cluster_nodes_dict = defaultdict(list)
        self.nodes_cluster_dict = {}

        self.node_not_in_cluster = [ i for i in range(self.num_vertices) ]

        initial_node = 0#np.random.randint(self.num_vertices)
        self.center_nodes.append(initial_node)


        #self.node_not_in_cluster.remove(initial_node)

        for node in self.node_not_in_cluster:
            if self.shortest_distance[initial_node][node] <= self.radius * distance_ratio:
                self.cluster_nodes_dict[initial_node].append(node)
                self.nodes_cluster_dict[node] = initial_node

        for node in self.cluster_nodes_dict[initial_node]:
            self.node_not_in_cluster.remove(node)

        while len(self.node_not_in_cluster) > 0:
            sampled_vertex = 0
            largest_distance, _ = self.closest_distance(self.node_not_in_cluster[sampled_vertex],self.center_nodes)
            largest_node = self.node_not_in_cluster[sampled_vertex]



            for tmp_node in self.node_not_in_cluster:
                tmp_distance, _ = self.closest_distance(tmp_node,self.center_nodes)
                if tmp_distance > largest_distance:
                    largest_distance = tmp_distance
                    largest_node = tmp_node


            self.center_nodes.append(largest_node)

            for node in self.node_not_in_cluster:
                if self.shortest_distance[largest_node][node] <= self.radius * distance_ratio:
                    self.cluster_nodes_dict[largest_node].append(node)
                    self.nodes_cluster_dict[node] = largest_node


            for node in self.cluster_nodes_dict[largest_node]:
                self.node_not_in_cluster.remove(node)

        self.K = len(self.center_nodes)

        cluster_size_list = [ len(self.cluster_nodes_dict[center_node]) for center_node in self.center_nodes ]

        tmp_indices = np.argsort(-np.array(cluster_size_list)).tolist()

        self.center_nodes = [ self.center_nodes[index] for index in tmp_indices]

        print(' Size_list = {} '.format([ len(self.cluster_nodes_dict[center_node]) for center_node in self.center_nodes ]))
        print('Distance ratio = {0}, K = {1}'.format(distance_ratio,self.K))


    def minimum_spanning_tree(self,Terminals):
        num_Terminals = len(Terminals)
        MST = np.zeros(shape=(num_Terminals,num_Terminals),dtype= np.int_)
        dict_terminals = {}
        for i,u in enumerate(Terminals):
            dict_terminals[u] = i
        for u in Terminals:
            for v in Terminals:
                MST[dict_terminals[u]][dict_terminals[v]] = self.shortest_distance[u][v]

        MST = csgraph.minimum_spanning_tree(csgraph=csr_matrix(MST)).toarray().astype(int)

        MST_cost = MST.sum().sum()




        MST_path = {}
        MST_path_cost = {}

        dist_matrix, predecessors = csgraph.shortest_path(csgraph=csr_matrix(MST),directed=False,return_predecessors=True)

        def get_path(dist_matrix,predecessors,u,v,Teriminals):
            path = [v]
            #path_cost = 0
            while path[0]!=u:
                x = path[0]
                y = predecessors[u][x]
                #path_cost += self.shortest_distance[x][y]
                path.insert(0,y)
            path_cost = dist_matrix[u][v]
            for i,node in enumerate(path):
                path[i] = Teriminals[node]
            return path,path_cost

        for i,u in enumerate(Terminals[:-1]):
            for v in Terminals[i+1:]:

                MST_path[(u,v)],MST_path_cost[(u,v)] = get_path(dist_matrix,predecessors,dict_terminals[u],dict_terminals[v],Terminals)

                MST_path[(v,u)] = MST_path[(u,v)].copy()
                MST_path[(v,u)].reverse()
                MST_path_cost[(v,u)] = MST_path_cost[(u,v)]


        return MST_cost,MST_path,MST_path_cost

    def greedy_algo(self,Terminals):
        greedy_cost = 0
        greedy_solution = []
        for i,u in enumerate(Terminals):
            if i > 0:
                greedy_cost += min([self.shortest_distance[x][u] for x in Terminals[:i]  ])

        return greedy_cost

    def predictive_algo(self,Terminals,Predicted_Terminals ):
        if len(Predicted_Terminals) == 0:
            return self.greedy_algo(Terminals)

        _,predicted_MST_path,predicted_MST_path_cost = self.minimum_spanning_tree(Predicted_Terminals)

        cost = 0
        solution = []
        connected_predicted_terminals = []
        for i,u in enumerate(Terminals):
            if u in connected_predicted_terminals:
                continue
            if i > 0:
                if u not in Predicted_Terminals or len(connected_predicted_terminals) == 0:
                    cost += min([self.shortest_distance[x][u] for x in Terminals[:i] +connected_predicted_terminals ])
                    if u in Predicted_Terminals:
                        connected_predicted_terminals.append(u)
                else:
                    connected_predicted_path_cost_list = [ predicted_MST_path_cost[(u,x)] for x in connected_predicted_terminals ]
                    min_cost = min(connected_predicted_path_cost_list)
                    cost += min_cost
                    min_index = connected_predicted_path_cost_list.index(min_cost)
                    min_x = connected_predicted_terminals[min_index]
                    path = predicted_MST_path[(u,min_x)]
                    connected_predicted_terminals += path


            else:
                if u in Predicted_Terminals:
                    connected_predicted_terminals.append(u)

        return cost


    def clever_predictive_algo(self, Terminals ,Predicted_Terminals,lam = 2):
        if len(Predicted_Terminals) == 0:
            return self.greedy_algo(Terminals)
        _,predicted_MST_path,predicted_MST_path_cost = self.minimum_spanning_tree(Predicted_Terminals)

        cost = 0
        solution = []
        connected_predicted_terminals = []
        cost_predicted_terminals = defaultdict(int)
        predicted_terminal_path = defaultdict(list)
        for i,u in enumerate(Terminals):
            if u in connected_predicted_terminals:
                cost += cost_predicted_terminals[u]
                tmp_flag = 0
                cost_u = cost_predicted_terminals[u]
                for node in predicted_terminal_path[u]:
                    if node == u:
                        tmp_flag = 1

                    if tmp_flag == 0:
                        cost_predicted_terminals[node] = 0
                    else:
                        cost_predicted_terminals[node] -= cost_u

                continue
            if i > 0:
                if u not in Predicted_Terminals or len(connected_predicted_terminals) == 0:

                    greedy_cost = min([self.shortest_distance[x][u] for x in Terminals[:i]  ])

                    predicted_cost_list = [self.shortest_distance[x][u] for x in connected_predicted_terminals  ]
                    if len(predicted_cost_list) == 0:
                        cost += greedy_cost
                    else:
                        predicted_cost = min( predicted_cost_list)

                        if greedy_cost <= predicted_cost:
                            cost += greedy_cost
                        else:
                            cost += predicted_cost
                            min_predicted_index = predicted_cost_list.index(predicted_cost)
                            min_predicted_node = connected_predicted_terminals[min_predicted_index]

                            if min_predicted_node not in Terminals[:i]:
                                cost += cost_predicted_terminals[min_predicted_node]

                                tmp_flag = 0
                                cost_min_x = cost_predicted_terminals[min_predicted_node]
                                for node in predicted_terminal_path[min_predicted_node]:
                                    if node == min_predicted_node:
                                        tmp_flag = 1

                                    if tmp_flag == 0:
                                        cost_predicted_terminals[node] = 0
                                    else:
                                        cost_predicted_terminals[node] -= cost_min_x

                    if u in Predicted_Terminals:
                        connected_predicted_terminals.append(u)
                        cost_predicted_terminals[u] = 0
                else:
                    connected_predicted_path_cost_list = [ predicted_MST_path_cost[(u,x)] for x in connected_predicted_terminals ]
                    min_cost = min(connected_predicted_path_cost_list)

                    min_index = connected_predicted_path_cost_list.index(min_cost)
                    min_x = connected_predicted_terminals[min_index]



                    path = predicted_MST_path[(u,min_x)]
                    len_path = len(path)


                    min_greedy_cost = self.shortest_distance[Terminals[0]][u]
                    min_greedy_terminal = Terminals[0]

                    for x in Terminals[:i]:
                        tmp_greedy_cost = self.shortest_distance[x][u]
                        if tmp_greedy_cost < min_greedy_cost:
                            min_greedy_cost = tmp_greedy_cost
                            min_greedy_terminal = x
                    #min([ for x in Terminals[:i]  ])

                    connected_predicted_terminals.append(u)
                    cost_predicted_terminals[u] = 0

                    min_predicted_greedy_cost = min([self.shortest_distance[x][u] for x in Terminals[:i] if x in Predicted_Terminals  ])

                    tmp_cost = 0

                    feasible_flag = False

                    new_added_predicted_terminals_list = []
                    for index in range(len_path-1):
                        tmp_u = path[index]
                        tmp_v = path[index+1]
                        if tmp_cost + self.shortest_distance[tmp_u][tmp_v] <= 2*min_predicted_greedy_cost:#min_greedy_cost:
                            tmp_cost += self.shortest_distance[tmp_u][tmp_v]
                            if tmp_v in connected_predicted_terminals:
                                feasible_flag = True
                                break
                            else:
                                connected_predicted_terminals.append(tmp_v)
                                cost_predicted_terminals[tmp_v] = tmp_cost
                                new_added_predicted_terminals_list.append(tmp_v)

                        else:
                            break
                    for node in new_added_predicted_terminals_list:
                        predicted_terminal_path[node] = new_added_predicted_terminals_list.copy()
                    #cost += tmp_cost

                    if feasible_flag == False:
                        #cost += min_greedy_cost

                        greedy_cost = min([self.shortest_distance[x][u] for x in Terminals[:i]  ])
                        #fix bug
                        predicted_cost_list = [self.shortest_distance[x][u] for x in connected_predicted_terminals  if x not in new_added_predicted_terminals_list+[u]]
                        if len(predicted_cost_list) == 0:
                            cost += greedy_cost
                        else:
                            predicted_cost = min( predicted_cost_list)

                            if greedy_cost <= predicted_cost:
                                cost += greedy_cost
                            else:
                                cost += predicted_cost
                                min_predicted_index = predicted_cost_list.index(predicted_cost)
                                min_predicted_node = connected_predicted_terminals[min_predicted_index]

                                if min_predicted_node not in Terminals[:i]:
                                    cost += cost_predicted_terminals[min_predicted_node]

                                    tmp_flag = 0
                                    cost_min_x = cost_predicted_terminals[min_predicted_node]
                                    for node in predicted_terminal_path[min_predicted_node]:
                                        if node == min_predicted_node:
                                            tmp_flag = 1

                                        if tmp_flag == 0:
                                            cost_predicted_terminals[node] = 0
                                        else:
                                            cost_predicted_terminals[node] -= cost_min_x


                    else:
                        cost += tmp_cost

                        for node in new_added_predicted_terminals_list:
                            cost_predicted_terminals[node] = 0

                        if min_x not in Terminals[:i]:
                            cost += cost_predicted_terminals[min_x]

                            tmp_flag = 0
                            cost_min_x = cost_predicted_terminals[min_x]
                            for node in predicted_terminal_path[min_x]:
                                if node == min_x:
                                    tmp_flag = 1

                                if tmp_flag == 0:
                                    cost_predicted_terminals[node] = 0
                                else:
                                    cost_predicted_terminals[node] -= cost_min_x

                    #connected_predicted_terminals += path
                    #cost += min_cost


            else:
                if u in Predicted_Terminals:
                    connected_predicted_terminals.append(u)
                    cost_predicted_terminals[u] = 0

        return cost




















