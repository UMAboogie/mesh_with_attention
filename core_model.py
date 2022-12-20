# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Core learned graph net model."""

import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf
import numpy as np

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features','sender_features','reciever_features', 'edge_sets'])


class GraphNetBlock(snt.AbstractModule):
  """Multi-Edge Interaction Network with residual connections."""

  def __init__(self, model_fn, latent_size, num_head, adj, name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self._model_fn = model_fn
    self._latent_size = latent_size
    self._num_head = num_head
    self._adj = adj

  def _make_ffn(self, num_layers, with_bias=False, layer_norm=True, activate_final=False):
    """Builds an FFN."""
    widths = [self.latent_size] * num_layers + [self.latent_size]
    network = snt.nets.MLP(widths, activate_final=activate_final, with_bias=with_bias)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  """
  def _update_edge_features0(self, node_features, edge_set):
    sender_features = tf.gather(node_features, edge_set.senders)
    receiver_features = tf.gather(node_features, edge_set.receivers)
    features = [sender_features, receiver_features, edge_set.features]
    with tf.variable_scope(edge_set.name+'_edge_fn'):
      return self._model_fn()(tf.concat(features, axis=-1))
  """

    
  def _update_all_features(self, node_features, reciever_features, sender_features):
    #linear projection of features 
    proj_r = self._make_ffn(num_layers=1)(reciever_features)
    proj_s = self._make_ffn(num_layers=1)(sender_features)
    proj_m = self._make_ffn(num_layers=1)(node_features)

    #update edge features 
    next_reciever_features = proj_r + proj_m
    next_sender_features = proj_s + proj_m

    #residual connection
    next_reciever_features += reciever_features
    next_sender_features += sender_features

    num_nodes = node_features.get_shape[0]
    latent_size = node_features.get_shape[1]
    assert latent_size % self.num_head == 0

    proj_q = self._make_ffn(num_layers=1, with_bias=False, activate_final=True)(node_features)

    # N*latent_size -> N*num_head*(latent_size // num_head)
    proj_q_head = tf.reshape(proj_q, [num_nodes, self._num_head, latent_size // self._num_head])
    reciever_features_head = tf.reshape(next_reciever_features, [num_nodes, self._num_head, latent_size // self._num_head])
    sender_features_head = tf.reshape(next_sender_features, [num_nodes, self._num_head, latent_size // self._num_head])

    # N*num_head*(latent_size // num_head) -> num_head*N*(latent_size // num_head)
    head_proj_q = tf.transpose(proj_q_head, perm=[1,0,2])
    head_reciever_features = tf.transpose(reciever_features_head, perm=[1,0,2])
    head_sender_features = tf.transpose(sender_features_head, perm=[1,0,2])

    QR = tf.einsum('ijk,ijk->ij', head_proj_q, head_reciever_features)
    QR_3d = tf.expand_dims(QR, 2)
    attn_before_sm = tf.tile(QR_3d, [1, 1, num_nodes]) + tf.einsum('ijk,ilk->ijl', head_proj_q, head_sender_features)

    mask = tf.expand_dims(self._adj, 0)
    mask_head = tf.tile(mask, [self._num_head, 1, 1])
    attn_before_sm_with_adj = (1 - tf.cast(mask_head, tf.float32))*(-np.inf) + tf.cast(mask_head, tf.float32)*attn_before_sm
    scaled_attn_before_sm = attn_before_sm_with_adj / tf.math.sqrt(latent_size // self._num_head)
    attn_score = tf.math.softmax(scaled_attn_before_sm, dim=2)
    attn_dropout = tf.nn.dropout(attn_score)
    head_aggr_sender = tf.einsum('ijk,ikl->ijl', attn_dropout, head_sender_features)
    aggr_sender_head = tf.transpose(head_aggr_sender, perm=[1,0,2])
    aggr_sender = tf.reshape(aggr_sender_head, [num_nodes, latent_size])
    next_node_features = next_reciever_features + aggr_sender

    proj_node = self._make_ffn(num_layers=1, with_bias=True, activate_final=True)(next_node_features)
    proj_r = self._make_ffn(num_layers=1, with_bias=True, activate_final=True)(next_reciever_features)
    proj_s = self._make_ffn(num_layers=1, with_bias=True, activate_final=True)(next_sender_features)

    return proj_node, proj_r, proj_s



  """
  def _update_node_features0(self, node_features, edge_sets):
    Aggregrates edge features, and applies node function.
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                   edge_set.receivers,
                                                   num_nodes))
    with tf.variable_scope('node_fn'):
      return self._model_fn()(tf.concat(features, axis=-1))
  """

  def _build(self, graph):
    """Applies GraphNetBlock and returns updated MultiGraph."""
    # apply node function
    new_node_features, new_reciever_features, new_sender_features = self._update_all_features(graph.node_features, graph.reciever_features, graph.sender_features)

    # add residual connections
    new_node_features += graph.node_features

    return MultiGraph(new_node_features, new_reciever_features, new_sender_features, graph.edge_sets)


class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode GraphNet model."""

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               num_head,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    self._num_head = num_head

  def _make_mlp(self, output_size, layer_norm=True):
    """Builds an MLP."""
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _make_adj(self, graph):
    #note that a_ij = 1 (if edge j->i exists)
    num_nodes = graph.node_features.get_shape[0]
    for edge_set in graph.edge_sets:
      edge_list = tf.stack([edge_set.senders, edge_set.receivers], axis=1)
      A = tf.tensor_scatter_nd_update(tf.zeros((num_nodes, num_nodes), dtype=tf.int64), edge_list, tf.repeat(tf.cast(1, tf.int64), tf.shape(edge_list)[0]))
    return A


  def _encoder(self, graph):
    """Encodes node and edge features into latent features."""
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size)(graph.node_features)
      reciever_latents = self._make_mlp(self._latent_size)(node_latents)
      sender_latents = self._make_mlp(self._latent_size)(node_latents)
      new_edges_sets = []
      """
      for edge_set in graph.edge_sets:
        latent = self._make_mlp(self._latent_size)(edge_set.features)
        new_edges_sets.append(edge_set._replace(features=[]))
      """
    return MultiGraph(node_latents, reciever_latents, sender_latents, new_edges_sets)

  def _decoder(self, graph):
    """Decodes node features from graph."""
    with tf.variable_scope('decoder'):
      decoder = self._make_mlp(self._output_size, layer_norm=False)
      return decoder(graph.node_features)

  def _build(self, graph):
    """Encodes and processes a multigraph, and returns node features."""
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    latent_graph = self._encoder(graph)
    adj = self._make_adj(latent_graph)
    for _ in range(self._message_passing_steps):
      latent_graph = GraphNetBlock(model_fn, self._latent_size, self._num_head, adj)(latent_graph)
    return self._decoder(latent_graph)
