import tensorflow as tf
import numpy as np
from RL_brain_test import DeepQNetwork

sess1 = tf.Session()
saver = tf.train.import_meta_graph('./model_6_1/model_6_1.meta')
saver.restore(sess1, tf.train.latest_checkpoint('./model_6_1'))

graph = tf.get_default_graph()
x1 = graph.get_tensor_by_name('s:0')
y1 = graph.get_tensor_by_name('eval_net/l3/output:0')

#get inut for testing
with open('x_6_1.csv') as x:
        testing_data = np.fromstring(x.readline(), dtype=float, sep=',')[np.newaxis]
        print('testing_data is of shape ' + str(testing_data.shape))

#calculate outpout for testing input
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_out = sess.run(y1, feed_dict={x1: testing_data})
    print("shape of out put is " + str(y_out.shape))
    print('recommendation is ' + str(y_out))

sess1.close()