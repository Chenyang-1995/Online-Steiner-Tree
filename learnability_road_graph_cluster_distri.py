
import real_graph
import numpy as np
import random
import math



def test_road_graph_cluster_distri(original_num_training_instances,num_terminals=2000,distance_ratio=0.1,num_t_per_cluster = 100):

    G = real_graph.graph()
    G.load_data(FILE_NAME='road_graph/graph_0.txt')
    G.ratio_threhold_center_cluster(distance_ratio=distance_ratio)
    num_vertices = G.num_vertices
    total_performance_list = []

    total_error_list = []

    num_sampled_cluster = num_terminals // num_t_per_cluster



    original_large_center_list = [ x for x in G.center_nodes if len(G.cluster_nodes_dict[x]) >= num_t_per_cluster ]



    for num_training_instances in original_num_training_instances:
        num_training_instances = [num_training_instances]
        num_instances = max(num_training_instances)+1
        performance_list = []
        error_list = []
        for _ in range(10):

                if num_sampled_cluster <= len(original_large_center_list):
                    gap1 = num_sampled_cluster//2
                    gap2 = num_sampled_cluster//2

                    large_center_list = original_large_center_list[:gap1] + list(np.random.choice(original_large_center_list[gap1:-gap2],num_sampled_cluster-gap1-gap2,replace=False) ) + original_large_center_list[-gap2:]
                    print([len(G.cluster_nodes_dict[x]) for x in large_center_list ])
                else:
                    large_center_list = original_large_center_list.copy()

                count_set = [ 0 for _ in range(num_vertices) ]
                for instance_id in range(num_instances):



                    if instance_id in num_training_instances:

                            random_terminals = []
                            for center_node in large_center_list:
                                random_terminals += list(np.random.choice(G.cluster_nodes_dict[center_node],num_t_per_cluster,replace=False))

                            predicted_terminals_set = []
                            theta_list = [0,0.2,0.4,0.6,0.8,1.0]
                            for theta in theta_list:
                                predicted_terminals = []
                                for node in range(num_vertices):
                                    if np.random.uniform(0,1) <= 1.0*count_set[node]/instance_id and 1.0*count_set[node]/instance_id > theta:
                                        predicted_terminals.append(node)
                                predicted_terminals_set.append(predicted_terminals)


                            Greedy_cost = G.greedy_algo(random_terminals)
                            #print('Greedy_cost = {}'.format(Greedy_cost))

                            theta_oapt_performance_list = []
                            theta_ioapt_performance_list = []
                            for predicted_terminals in predicted_terminals_set:
                                tmp_random_terminals = train_random_terminals.copy()


                                tmp_Greedy_cost = G.greedy_algo(tmp_random_terminals)
                                theta_oapt_performance_list.append(1.0*G.predictive_algo(Terminals=tmp_random_terminals,Predicted_Terminals=predicted_terminals)/tmp_Greedy_cost)
                                theta_ioapt_performance_list.append(1.0*G.clever_predictive_algo(Terminals=tmp_random_terminals,Predicted_Terminals=predicted_terminals)/tmp_Greedy_cost)
                                print('select theta iter = {}'.format(len(theta_ioapt_performance_list)))

                            oapt_theta_index = theta_oapt_performance_list.index(min(theta_oapt_performance_list))


                            oapt_wrong_pred = len([1 for x in predicted_terminals_set[oapt_theta_index] if x not in random_terminals])
                            print('OAPT theta = {0}, OAPT_eta = {1}'.format(theta_list[oapt_theta_index],oapt_wrong_pred))

                            ioapt_theta_index = theta_ioapt_performance_list.index(min(theta_ioapt_performance_list))

                            ioapt_wrong_pred = len([1 for x in predicted_terminals_set[ioapt_theta_index] if x not in random_terminals])

                            print('IOAPT theta = {0}, IOAPT_eta = {1}'.format(theta_list[ioapt_theta_index],ioapt_wrong_pred))

                            Predictive_cost = 1.0*G.predictive_algo(Terminals=random_terminals,Predicted_Terminals=predicted_terminals_set[oapt_theta_index])/Greedy_cost


                            Clever_Predictive_cost = 1.0*G.clever_predictive_algo(Terminals=random_terminals,Predicted_Terminals=predicted_terminals_set[ioapt_theta_index])/Greedy_cost



                            if instance_id % 1 == 0:


                                logging_str = "New_instance  = {}\n".format(instance_id)

                                logging_str += 'Predictive_cost = {}\n'.format(Predictive_cost)

                                logging_str += 'Clever_Predictive_cost = {}\n'.format(Clever_Predictive_cost)


                                print (logging_str)



                                performance_list.append([ Predictive_cost,Clever_Predictive_cost ])
                                error_list.append([oapt_wrong_pred,ioapt_wrong_pred])

                                #print(performance_list)


                    train_random_terminals = []
                    for center_node in large_center_list:
                        train_random_terminals += list(np.random.choice(G.cluster_nodes_dict[center_node],num_t_per_cluster,replace=False))


                    for node in train_random_terminals:
                        count_set[node] += 1


        total_performance_list.append(performance_list)
        print('-'*80)
        print('Instance id = {}'.format(num_training_instances))
        print("Performance:")
        print(total_performance_list)
        print('Pred error:')
        print(total_error_list)






if __name__=='__main__':



    num_training_instances = [ int(math.pow(2,x)) for x in range(13)]

    num_terminals_per_cluster = 100

    test_road_graph_cluster_distri(original_num_training_instances=num_training_instances,num_t_per_cluster=num_terminals_per_cluster)