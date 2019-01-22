#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import six
import tensorflow as tf

from tensorflow.contrib import tensorrt as trt


def main():
    with tf.Graph().as_default(), tf.Session() as session:
        batch_size = 1
        input = tf.placeholder(shape=(batch_size, 256, 256, 3), dtype=tf.float32, name='input')
        trunk = _make_fusable_subgraph(input)
        trunk = tf.where(trunk > 5, trunk, -trunk)
        heads = [
            tf.identity(
                _make_fusable_subgraph(trunk),
                name='output_{}'.format(i)
            )
            for i in six.moves.xrange(10)
        ]

        graph_def = session.graph.as_graph_def()

    output_node_names = [_get_node_name(head) for head in heads]

    graph_def = trt.create_inference_graph(
        input_graph_def=graph_def,
        outputs=output_node_names,
        max_batch_size=batch_size,
        max_workspace_size_bytes=10**9,
        precision_mode='INT8',
        minimum_segment_size=10
    )

    with tf.Graph().as_default(), tf.Session() as session:
        tf.import_graph_def(
            graph_def,
            name='',
        )
        input = session.graph.get_tensor_by_name('input:0')
        heads = [session.graph.get_tensor_by_name('output_{}:0'.format(i)) for i in six.moves.xrange(10)]

        for i in six.moves.xrange(10):
            session.run(heads, feed_dict={
                input: np.random.random_integers(low=0, high=255, size=(batch_size, 256, 256, 3))
            })

    graph_def = trt.calib_graph_to_infer_graph(graph_def)

    serialized_graph_def = graph_def.SerializeToString()
    assert 'my_trt_op' in serialized_graph_def

    with tf.gfile.GFile('graph.pb', 'wb') as f:
        f.write(serialized_graph_def)


def _make_fusable_subgraph(net):
    for i in six.moves.xrange(10):
        in_channels = net.get_shape()[-1].value
        out_channels = 16
        net = tf.nn.conv2d(
            net,
            filter=np.random.rand(3, 3, in_channels, out_channels),
            strides=(1, 1, 1, 1),
            padding='SAME'
        )
        net = tf.nn.relu(net)
    return net


def _get_node_name(tensor):
    assert tensor.name.endswith(':0')
    return tensor.name[:-len(':0')]


if __name__ == '__main__':
    main()
