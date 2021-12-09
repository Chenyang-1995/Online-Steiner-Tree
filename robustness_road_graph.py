
import real_graph
import numpy as np






def test_fix_accuracy(ratio,num_terminals=2000,num_instances=10):

    G = real_graph.graph()
    G.load_data(FILE_NAME='road_graph/graph_0.txt')
    num_vertices = G.num_vertices


    num_accurate_predictions = int(num_terminals*ratio)
    num_wrong_predictions = num_terminals - num_accurate_predictions
    print('accurate_num = {0}, num_wrong_pred = {1}'.format(num_accurate_predictions,num_wrong_predictions))


    performance_list = []
    for instance_id in range(num_instances):

        random_terminals = np.random.choice(num_vertices,num_terminals,replace=False).tolist()
        non_terminal_set = [ x for x in range(num_vertices) if x not in random_terminals]

        predicted_terminals = list(np.random.choice(random_terminals,num_accurate_predictions,replace=False)) \
                                + list(np.random.choice(non_terminal_set,num_wrong_predictions,replace=False))

        pred_error = 1.0*sum([1 for x in random_terminals if x not in predicted_terminals]) / len(random_terminals)
        logging_str = "New_instance  = {}\n".format(instance_id)
        logging_str += 'Pred_Error = {}\n'.format(pred_error)
        print(logging_str)

        Greedy_cost = G.greedy_algo(random_terminals)



        Predictive_cost = 1.0*G.predictive_algo(Terminals=random_terminals,Predicted_Terminals=predicted_terminals)/Greedy_cost


        Clever_Predictive_cost = 1.0*G.clever_predictive_algo(Terminals=random_terminals,Predicted_Terminals=predicted_terminals)/Greedy_cost

        if True:
            pred_error = 1.0*sum([1 for x in random_terminals if x not in predicted_terminals]) / len(random_terminals)
            logging_str = "New_instance  = {}\n".format(instance_id)
            logging_str += 'Pred_Error = {}\n'.format(pred_error)
            logging_str = "Test_fix_accuracy {} ".format(ratio)
            logging_str += 'Instance {} '.format(instance_id)
            logging_str += 'Greedy_cost = {} '.format(Greedy_cost)
            logging_str += 'Predictive_cost = {} '.format(Predictive_cost)
            logging_str += 'Clever_Predictive_cost = {}'.format(Clever_Predictive_cost)
            print (logging_str)

            performance_list.append([ Predictive_cost,Clever_Predictive_cost ])



    return performance_list



if __name__=='__main__':


    #test_instance_set_real_graph(high_ratio=0.5)

    total_performance_list = []
    for r in [0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.8,1.0]:
        print('------------Start {}-----------'.format(r))
        total_performance_list.append(test_fix_accuracy(ratio=r) )
        print(total_performance_list)
        print('------------End {}-----------'.format(r))