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

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features','sender_features','reciever_features', 'edge_sets'])


class GraphNetBlock(snt.AbstractModule):
  """Multi-Edge Interaction Network with residual connections."""

  def __init__(self, model_fn, latent_size, num_head, name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self._model_fn = model_fn
    self.latent_size = latent_size
    self.num_head = num_head

  def _make_ffn(self, num_layers, with_bias=False, layer_norm=True):
    """Builds an FFN."""
    widths = [self.latent_size] * num_layers + [self.latent_size]
    network = snt.nets.MLP(widths, activate_final=False, with_bias=with_bias)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _update_edge_features0(self, node_features, edge_set):
    """Aggregrates node features, and applies edge function."""
    sender_features = tf.gather(node_features, edge_set.senders)
    receiver_features = tf.gather(node_features, edge_set.receivers)
    features = [sender_features, receiver_features, edge_set.features]
    with tf.variable_scope(edge_set.name+'_edge_fn'):
      return self._model_fn()(tf.concat(features, axis=-1))

  def _update_edge_features(self, node_features, sender_features, reciever_features, edge_set):
    """Caluculates attention score."""
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

    return next_reciever_features, next_sender_features

    
  def _update_node_features(self, node_features, sender_features, reciever_features, edge_sets):
    num_nodes = node_features.shape[0]
    latent_size = node_features.shape[1]

    proj_q = self._make_ffn(latent_size=128, num_layers=1, output_size=128, with_bias=True)(node_features)

    # N*latent_size -> N*num_head*(latent_size // num_head)
    proj_q_head = tf.reshape(proj_q, [num_nodes, self.num_head, latent_size // self.num_head])
    reciever_features_head = tf.reshape(reciever_features, [num_nodes, self.num_head, latent_size // self.num_head])
    sender_features_head = tf.reshape(sender_features, [num_nodes, self.num_head, latent_size // self.num_head])

    # N*num_head*(latent_size // num_head) -> num_head*N*(latent_size // num_head)
    proj_q_head = tf.transpose(proj_q_head, perm=[1,0,2])
    reciever_features_head = tf.transpose(reciever_features_head, perm=[1,0,2])
    sender_features_head = tf.transpose(sender_features_head, perm=[1,0,2])

    QR = tf.einsum('ijk,ijk->ij', proj_q_head, reciever_features_head)
    QR = tf.reshape(QR, [QR.shape[0], QR.shape[1], 1])
    score_before_sm = tf.tile(QR, [1,1,num_nodes]) + tf.einsum('ijk,ilk->ijl', proj_q_head, sender_features_head)

    return

  def _update_node_features0(self, node_features, edge_sets):
    """Aggregrates edge features, and applies node function."""
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                   edge_set.receivers,
                                                   num_nodes))
    with tf.variable_scope('node_fn'):
      return self._model_fn()(tf.concat(features, axis=-1))

  def _build(self, graph):
    """Applies GraphNetBlock and returns updated MultiGraph."""

    # apply edge functions
    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated_features = self._update_edge_features(graph.node_features,
                                                    edge_set)
      new_edge_sets.append(edge_set._replace(features=updated_features))

    # apply node function
    new_node_features = self._update_node_features(graph.node_features,
                                                   new_edge_sets)

    # add residual connections
    new_node_features += graph.node_features
    new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)


class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode GraphNet model."""

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps

  def _make_mlp(self, output_size, layer_norm=True):
    """Builds an MLP."""
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _encoder(self, graph):
    """Encodes node and edge features into latent features."""
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size)(graph.node_features)
      reciever_latents = self._make_mlp(self._latent_size)(node_latents)
      sender_latents = self._make_mlp(self._latent_size)(node_latents)
      new_edges_sets = []
      for edge_set in graph.edge_sets:
        #latent = self._make_mlp(self._latent_size)(edge_set.features)
        new_edges_sets.append(edge_set._replace(features=[]))
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
    for _ in range(self._message_passing_steps):
      latent_graph = GraphNetBlock(model_fn)(latent_graph)
    return self._decoder(latent_graph)
