import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf



tf.enable_resource_variables()
tf.disable_eager_execution()
x = tf.constant([[1,2],[3,4]])
y = tf.tile(tf.reshape(x, [x.shape[0], x.shape[1], 1]), [1,1,4])

print(y.eval(session=tf.Session()))
