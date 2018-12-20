import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from RL_brain_test import DeepQNetwork

sess1 = tf.Session()
#saver = tf.train.import_meta_graph('./model_6_1/model_6_1.meta')
#saver.restore(sess1, tf.train.latest_checkpoint('./model_6_1'))
saver = tf.train.import_meta_graph('./elen6885/model/model_a_0_b_1/model_a_0_b_1.meta')
saver.restore(sess1, tf.train.latest_checkpoint('./elen6885/model/model_a_0_b_1'))

graph = tf.get_default_graph()
x1 = graph.get_tensor_by_name('s:0')
y1 = graph.get_tensor_by_name('eval_net/l3/output:0')

with open('./elen6885/data/x.csv') as x:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total = []
        i = 0
        for line in x:
            testing_data = np.fromstring(line, dtype=float, sep=',')[np.newaxis]
            y_out = sess.run(y1, feed_dict={x1: testing_data})
            print(y_out)
            y_max = np.amax(y_out)
            y_max = math.log10(y_max)
            #print("ymax is " + str(y_max))
            total.append(y_max)
            i += 1
            if i % 200 == 0:
                percent = int(i * 10000/ 550000)/100
                print(str(percent) + "% finished")
                break

        plt.plot(total)
        plt.ylabel("Rewards")
        plt.xlabel("State Space")
        plt.show()

sess1.close()

print('testing_data is of shape ' + str(testing_data.shape))
print("shape of out put is " + str(y_out.shape))
        #print('recommendation is ' + str(y_out))