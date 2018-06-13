# -*- coding: utf-8 -*-
import tensorflow as tf
import pickle
import tensorflow.contrib.slim as slim
import sys, os
import numpy as np
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops   #tensorflow/tensorflow/python/ops/ ?
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, \
    initializers, layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import array_ops
BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def batch_normalization_layer(input_layer, dimension,name):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    with tf.variable_scope(name):
        beta = tf.get_variable(name='beta',
                                shape = dimension, 
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0, tf.float32),
                                regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
        gamma = tf.get_variable(name='gamma',
                                shape = dimension, 
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0, tf.float32),
                                regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """Strided 2-D convolution with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.

    Note that

      net = conv2d_same(inputs, num_outputs, 3, stride=stride)

    is equivalent to

      net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=1,
      padding='SAME')
      net = subsample(net, factor=stride)

    whereas

      net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=stride,
      padding='SAME')

    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.

    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
      return layers_lib.conv2d(
                        inputs,
                        num_outputs,
                        [kernel_size,kernel_size],
                        stride=1,
                        rate=rate,
                        padding='SAME',
                        activation_fn=None,
                        scope=scope)
    
                        
    else:
      kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
      pad_total = kernel_size_effective - 1
      pad_beg = pad_total // 2
      pad_end = pad_total - pad_beg
      inputs = array_ops.pad(
          inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      return layers_lib.conv2d(
                        inputs,
                        num_outputs,
                        kernel_size,
                        stride=stride,
                        rate=rate,
                        padding='VALID',
                        activation_fn=None,
                        scope=scope)
            #  tf.layers.conv2d(inputs=inputs,
            #                     filters=num_outputs,
            #                     kernel_size=[kernel_size,kernel_size],
            #                     strides=[1, 1],
            #                     dilation_rate=(rate,rate),
            #                     padding='VALID',
            #                     activation=tf.nn.relu,
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name=scope)

def subsample(inputs, factor, name=None):
    """Subsamples the input along the spatial dimensions.

    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.

    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
      return inputs
    else:
      return layers.max_pool2d(inputs, [1, 1], stride=factor, scope=name)

def conv_bn(inputs,
            filters,
            kernel_size,
            stride,
            activation,        
            name):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=[kernel_size, kernel_size],
                                strides=[stride, stride],
                                padding='same',
                                activation=activation,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='conv')
        bn = batch_normalization_layer(conv, filters,name='bn')
    return conv
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               name,
               rate=1):
    with tf.variable_scope(name):
        depth_in = inputs.get_shape().as_list()[-1]
        conv1 = tf.layers.conv2d(inputs=inputs,
                                filters=depth_bottleneck,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                dilation_rate=(rate,rate),
                                padding='same',
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='conv1')
        bn1 = batch_normalization_layer(conv1, depth_bottleneck,name='bn1')

        conv2 = conv2d_same(inputs=tf.nn.relu(bn1), 
                            num_outputs=depth_bottleneck, 
                            kernel_size=3, 
                            stride=stride, 
                            rate=rate, scope='conv2') 
        '''
        conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None)
        Strided 2-D convolution with 'SAME' padding.
        depth_bottleneck: The depth of the bottleneck layers
        '''
        bn2 = batch_normalization_layer(conv2, depth_bottleneck,name='bn2')
    
        conv3 = tf.layers.conv2d(inputs=tf.nn.relu(bn2),
                                filters=depth,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                dilation_rate=(rate,rate),
                                padding='same',
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='conv3')
        bn3 = batch_normalization_layer(conv3, depth,name='bn3')
        shortcut = subsample(inputs, factor=stride, name='shortcut')
        output = tf.nn.relu(bn3+shortcut,name='output')

    return output

def resblock(inputs,
               depth,
               stride,
               name,
               rate=1):
    
    conv2 = conv2d_same(inputs=inputs, 
                        num_outputs=depth, 
                        kernel_size=3, 
                        stride=stride, 
                        rate=rate, scope=name+'/conv2') 
    '''
    conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None)
    Strided 2-D convolution with 'SAME' padding.
    depth_bottleneck: The depth of the bottleneck layers
    '''
    with tf.variable_scope(name):
        bn2 = batch_normalization_layer(conv2, depth,name='bn2')
        shortcut = inputs
        output = tf.nn.relu(bn2+shortcut,name='output')
    return output