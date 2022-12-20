import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf



tf.enable_resource_variables()
tf.disable_eager_execution()
#x = tf.constant([[1,2],[3,4]])
#y = tf.Variable(x)
#print(y.eval(session=tf.Session()))



#y = y[0,0].assign(7)
#print(y.eval(session = tf.Session()))
edge_list = tf.constant([[0,1],[1,2],[2,3],[0,2],[1,3],[1,0],[2,1],[3,2],[2,0],[3,1]])

num_nodes = 4
#adj = tf.Variable(tf.zeros([num_nodes, num_nodes], tf.int32), trainable=False)

"""
for edge in edge_list:
    adj = adj[edge[0],edge[1]].assign(1)
"""
#adj = tf.assign(adj[edge_list[0][0],edge_list[0][1]],1)
#loop_op = tf.while_loop(lambda i, adj:i < 4, lambda i, adj: (i+1, tf.assign(adj[edge_list[i,0],edge_list[i,1]],1)), (0, adj))
#adj = adj[edge_list[1][0],edge_list[1][1]].assign(1)


A = tf.tensor_scatter_nd_update(
    tf.zeros((num_nodes, num_nodes), dtype=tf.int64),
    edge_list,
    tf.repeat(tf.cast(1, tf.int64), tf.shape(edge_list)[0]),
)

mask = tf.expand_dims(A, 0)
mask_head = tf.tile(mask, [3, 1, 1])
mask_float = tf.cast(mask_head, tf.float32)
attn = tf.math.softmax(mask_float, dim=2)

place = tf.equal(A,1)
sess = tf.Session()
#sess.run(tf.initialize_all_variables())
print(attn.eval(session=sess))
