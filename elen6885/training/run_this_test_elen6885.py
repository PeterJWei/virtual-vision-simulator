from RL_brain_test import DeepQNetwork
import numpy as np
import tensorflow as tf
#from tensorflow.python.framework.graph_util import convert_variables_to_constants

'''
def build_and_save(people, devices, spaces, i):
    output_node = people * spaces + devices + spaces + people
    input_node = people + devices + spaces * 2 + 1

    print("\n\n\nNow building NN model\n\n\n")
    tf.reset_default_graph()

    RL = DeepQNetwork(output_node, input_node,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      replace_target_iter=200,
                      memory_size=1000,
                      #output_graph=True
                      e_greedy=0.9,
                      hidden_layer=i
                      )
    saver = tf.train.Saver()
    saver.save(RL.sess, "./model_6_19_{0}/model_6_19_{0}".format(i))
    print("\n\n\nModel saved in " + "model_6_19_{0}".format(i))
    print("The structure is: \n---Input layer nodes:{0}\n---Hidden layer nodes:{1}\n---Output layer nodes:{2}\n".format(input_node, i * 100, output_node))
'''


def run_maze():
    saver = tf.train.Saver()
    #action_total = 80

    #with open('x_6_1.csv') as x, open('y_6_1.csv') as y, open('z_6_1.csv') as z:
    with open('./elen6885/data/x.csv') as x, open('./elen6885/data/y_a_3_b_1.csv') as y, open('./elen6885/data/z.csv') as z, open('./elen6885/data/a.csv') as a:
        while True:
            states_string = x.readline()
            reward_string = y.readline()
            statesnxt_string = z.readline()
            action_string = a.readline().strip()

            if (not states_string) or (not reward_string) or (not statesnxt_string) or (not action_string):
                break

            action = int(action_string)
            observation = np.fromstring(states_string, dtype=float, sep=',')
            reward = np.fromstring(reward_string, dtype=float, sep=',')
            observation_ = np.fromstring(statesnxt_string, dtype=float, sep=',')

            RL.store_transition(observation, action, reward, observation_)

            print('action: ' + str(action))

    it = 0
    while it < 30000:
        RL.learn()
        print('step:' + str(it))
        it = it+1
    saver.save(RL.sess, "./elen6885/model/model_a_3_b_1/model_a_3_b_1")

    # Save as protobuf
    #minimal_graph = convert_variables_to_constants(RL.sess, RL.sess.graph_def, ["output"])

    #tf.train.write_graph(RL.sess.graph, './elen6885/model/model_a_3_b_1/', 'model_a_3_b1.pb', as_text=False)
    #tf.train.write_graph(minimal_graph, '.', 'minimal_graph.txt', as_text=True)




if __name__ == "__main__":
    action_total = 80
    n_people = 6
    n_cameras = 10
    n_features = (n_cameras * 4) + (n_people * 2)

    print("Staring Deep Q Network\n")
    RL = DeepQNetwork(action_total, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      replace_target_iter=200,
                      memory_size=1000,
                      # output_graph=True
                      e_greedy=0.9,
                      )
    #RL = DeepQNetwork(210, 75,
    #                  learning_rate=0.01,
    #                  reward_decay=0.9,
    #                  replace_target_iter=200,
    #                  memory_size=1000,
    #                  # output_graph=True
    #                  e_greedy=0.9,
    #                  )
    print("Instantiated Deep Q Network\n")
    run_maze()
    RL.plot_cost()
    '''
    people = 40
    devices = 80
    spaces = 20
    times = [1, 4, 10, 12, 16]
    for i in range(0, 1):
        people_i = people * times[i]
        devices_i = devices * times[i]
        spaces_i = spaces * times[i]
        build_and_save(people_i, devices_i, spaces_i, i+1)
    '''
