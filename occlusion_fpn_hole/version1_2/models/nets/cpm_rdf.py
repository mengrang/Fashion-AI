import tensorflow as tf
import pickle
import sys, os
import numpy as np
from functools import partial
import collections
from tensorflow.python.ops import variable_scope
sys.path.append('/home/lzhpc/home/gu/add occlusion and resnet+fpn/version1_3/models/nets/')
from cpn_m_utils import activation_summary,batch_normalization_layer,\
    subsample,conv_bn,bottleneck,conv2d_same,resblock
from tensorflow.contrib.layers.python.layers import utils


class CPM_Model(object):
    def __init__(self, total_num, input_size, heatmap_size, batch_size, stages, num_joints, img_type= 'RGB', is_training=True):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.num_joints = num_joints
        self.inference_type = 'Train'  # not be used
        self.batch_size_np = batch_size
        self.stage_loss_batch_hmindex = [0] * self.num_joints
        self.stage_loss_batch = [0] * self.batch_size_np
        self.total_num = total_num
        self.lateral_fms=[]
        self.sub_fms=[]
    
        if img_type == 'RGB':
            self.input_images = tf.placeholder(dtype=tf.float32,
                                               shape=(None, input_size, input_size, 3),
                                               name='input_placeholder')
        elif img_type == 'GRAY':
            self.input_images = tf.placeholder(dtype=tf.float32,
                                               shape=(None, input_size, input_size, 1),
                                               name='input_placeholder')

        self.cmap_placeholder = tf.placeholder(dtype=tf.float32,
                                               shape=(None, input_size, input_size, 1),
                                               name='cmap_placeholder')
        self.gt_hmap_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None, heatmap_size, heatmap_size, num_joints),     # not the same with cpm_hand，hand里头加了1，为何？背景？
                                                  name='gt_hmap_placeholder')
        self.train_weights_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None, num_joints),
                                                  name='train_weights_placeholder')
        self._build_model()

    def conv2d_transpose_strided(self,x, W, b, output_shape=None):
            
            if output_shape is None:
                output_shape = x.get_shape().as_list()
                output_shape[1] *= 2
                output_shape[2] *= 2
                output_shape[3] = W.get_shape().as_list()[2]
            
            conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding="SAME")
            return tf.nn.bias_add(conv, b)

    def _build_model(self):
        with tf.variable_scope('pooled_center_map'):
            self.center_map = tf.layers.average_pooling2d(inputs=self.cmap_placeholder,
                        pool_size=[9, 9],
                        strides=[8, 8],
                        padding='same',
                        name='center_map')
        with tf.variable_scope('sub_stages'):
            
            sub_unit_1 = tf.layers.conv2d(inputs=self.input_images,
                                filters=64,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(1,1),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_1')
            sub_unit_2 = resblock(inputs=sub_unit_1,
                            depth=64,
                            # depth_bottleneck=32,
                            stride=1,
                            name='sub_unit_2'
                            )
            sub_pool1 = tf.layers.average_pooling2d(inputs=sub_unit_2,
                        pool_size=[2, 2],
                        strides=[2,2],
                        padding='same',     
                        name='sub_pool1')
            
            sub_unit_3 = tf.layers.conv2d(inputs=sub_pool1,
                                filters=128,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(1,1),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_3')      
            sub_unit_4_1 = resblock(inputs=sub_unit_3,
                            depth=128,
                            # depth_bottleneck=32,
                            stride=1,
                            name='sub_unit_4_1'
                            )
            sub_unit_4_2 = tf.layers.conv2d(inputs=sub_unit_3,
                                filters=128,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(6,6),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_4_2')
            sub_unit_4_cat = tf.concat([sub_unit_4_1,sub_unit_4_2],axis=3)
            sub_pool2 = tf.layers.average_pooling2d(inputs= sub_unit_4_cat,
                        pool_size=[2, 2],
                        strides=[2,2],
                        padding='same',     
                        name='sub_pool2')
            
            sub_unit_5 = tf.layers.conv2d(inputs=sub_pool2,
                                filters=256,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(1,1),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_5')         
            sub_unit_6 = resblock(inputs=sub_unit_5,
                                depth=256,
                                # depth_bottleneck=64,
                                stride=1,
                                name='sub_unit_6'
                                )   
            sub_unit_7 = resblock(inputs=sub_unit_6,
                            depth=256,
                            # depth_bottleneck=64,
                            stride=1,
                            name='sub_unit_7'
                            )
            sub_unit_8_1 = tf.layers.conv2d(inputs=sub_unit_7,
                                filters=128,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                dilation_rate=(1,1),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_8_1')
            sub_unit_8_2 = tf.layers.conv2d(inputs=sub_unit_7,
                                filters=128,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(6,6),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_8_2')
            sub_unit_8_3 = tf.layers.conv2d(inputs=sub_unit_7,
                                filters=128,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(12,12),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_8_3')
            sub_unit_8_4 = tf.layers.conv2d(inputs=sub_unit_7,
                                filters=128,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(18,18),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_8_4')
            sub_unit_8_cat = tf.concat([sub_unit_8_1,sub_unit_8_2,sub_unit_8_3,sub_unit_8_4],axis=3)
            sub_pool3 = tf.layers.average_pooling2d(inputs=sub_unit_8_cat,
                        pool_size=[2, 2],
                        strides=[2,2],
                        padding='same',     
                        name='sub_pool3')

            sub_unit_9 = tf.layers.conv2d(inputs=sub_pool3,
                                filters=512,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(1,1),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_9')
            sub_unit_10 = resblock(inputs=sub_unit_9,
                            depth=512,
                            # depth_bottleneck=64,
                            stride=1,
                            name='sub_unit_10')
            sub_unit_11= tf.layers.conv2d(inputs=sub_unit_10,
                                filters=256,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                dilation_rate=(1,1),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sub_unit_11') 
            sub_unit_12 = resblock(inputs=sub_unit_11,
                            depth=256,
                            # depth_bottleneck=64,
                            stride=1,
                            name='sub_unit_12')
            sub_unit_13 = resblock(inputs=sub_unit_12,
                            depth=256,
                            # depth_bottleneck=64,
                            stride=1,
                            name='sub_unit_13')
            sub_unit_14 = resblock(inputs=sub_unit_13,
                            depth=256,
                            # depth_bottleneck=64,
                            stride=1,
                            name='sub_unit_14')
            sub_pool4 = tf.layers.average_pooling2d(inputs=sub_unit_14,
                        pool_size=[2, 2],
                        strides=[2,2],
                        padding='same',
                        name='sub_pool4')

            
            '''FPN'''

            c3 = sub_pool4  # channel = 256, size = 32
            c2 = sub_pool3  # channel = 256, size = 64
            c1 = sub_pool2  # channel = 128, size = 128
            sub_fms =[c1,c2,c3]
            lateral_fms = []
            for i,fm in enumerate(sub_fms):
                c_out = tf.layers.conv2d(inputs=fm,
                                filters=64,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                dilation_rate=(1,1),
                                padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='c{}_out'.format(i+1))
                lateral_fms.append(c_out)
            # p3_2 = tf.image.resize_bilinear(lateral_fms[-1], (64,64), name='p3_2')
            p3_2 = self.conv2d_transpose_strided(x=lateral_fms[-1],
                                 W=tf.get_variable(name="W_p3", shape=[3, 3, 64,64]),
                                 b=tf.get_variable(name="b_p3", shape=[64]),
                                 output_shape=tf.shape(lateral_fms[1]))   #size=64
            p2 = tf.add(p3_2,lateral_fms[1],name="p2")
            # p2_2 = tf.image.resize_bilinear(p2,(128,128),name='p2_2')
            p2_2 = self.conv2d_transpose_strided(x=p2,
                                                 W=tf.get_variable(name="W_p4", shape=[3, 3, 64, 64]),
                                                 b=tf.get_variable(name="b_p4", shape=[64]),
                                                 output_shape=tf.shape(lateral_fms[0]))  # size=128

            p1 = tf.add(p2_2,lateral_fms[0],name="p1")
            # p3_4 = tf.image.resize_bilinear(p3_2,(128,128),name='p3_4')
            p3_4 = self.conv2d_transpose_strided(x=p3_2,
                                         W=tf.get_variable(name="W_p3_4", shape=[3, 3, 64,64]),
                                         b=tf.get_variable(name="b_p3_4", shape=[64]),
                                         output_shape=tf.shape(lateral_fms[0]))   #size=64
            p_cat = tf.concat([p1,p2_2,p3_4],axis=3)

            # stride = 2, change size=128 to 64
                
            sub_unit_15 = tf.layers.conv2d(inputs=p_cat,
                                    filters=128,
                                    kernel_size=[3, 3],
                                    strides=[2, 2],
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='sub_unit_15')      # channel = 128, size = 64

            self.sub_stage_img_feature = tf.layers.conv2d(inputs=sub_unit_15,
                                    filters=128,
                                    kernel_size=[3, 3],
                                    strides=[1, 1],
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='sub_stage_img_feature')

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                    filters=512,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='conv1')

            self.stage_heatmap.append(tf.layers.conv2d(inputs=conv1,
                                    filters=self.num_joints,     # not the same with cpm_hand，hand里头加了1，为何？背景？
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    padding='same',      # not the same with cpm_hand，padding方法不一样
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='stage_heatmap'))

        for stage in range(2, self.stages + 1):
            self._middle_conv(stage)

    def _middle_conv(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            self.current_featuremap = tf.concat([self.stage_heatmap[stage - 2],
                                                 self.sub_stage_img_feature,
                                                 self.center_map],
                                                axis=3)

            mid_conv1 = tf.layers.conv2d(inputs=self.current_featuremap,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv1')

            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv2')

            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv3')

            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv4')

            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv5')

            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv6')

            self.current_heatmap = tf.layers.conv2d(inputs=mid_conv6,
                                                    filters=self.num_joints,
                                                    kernel_size=[1, 1],
                                                    strides=[1, 1],
                                                    padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='mid_conv7')

            self.stage_heatmap.append(self.current_heatmap)

    def build_loss(self, lr, lr_decay_rate, lr_decay_step, optimizer='Adam'):
        self.gt_heatmap = self.gt_hmap_placeholder
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

        # 计算每个stage的loss
        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        # 计算总loss
        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=self.optimizer)
        self.merged_summary = tf.summary.merge_all()

    # new learning rate setting method
    def build_loss2(self, optimizer='Adam'):
        self.gt_heatmap = self.gt_hmap_placeholder
        self.total_loss = 0
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

        # 计算每个stage的loss
        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        # 计算总loss
        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            step = tf.cast(self.global_step, dtype=tf.float64)
            # new learning rate setting method
            epcho = step * 16.0 / 10000
            new_lr = tf.cond(tf.less(epcho, tf.constant(20.0, dtype=tf.float64)),
                     lambda: 0.0006 / 20.0 * epcho,
                     lambda: tf.cond(tf.less(epcho, tf.constant(60.0, dtype=tf.float64)),
                             lambda: tf.constant(0.0006, dtype=tf.float64),
                             lambda: 0.0006 - ((epcho - 60) / (100.0 - 60.0) * 0.0006)
                     ))
            self.lr = new_lr
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=self.optimizer)
        self.merged_summary = tf.summary.merge_all()

    # new learning rate setting method
        # new learning rate setting method

    def build_loss3(self, optimizer='Adam'):
        self.gt_heatmap = self.gt_hmap_placeholder
        self.train_weights = self.train_weights_placeholder
        self.total_loss = 0
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

        # 计算每个stage的loss, weighted l2 loss
        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss_batch = [0] * self.batch_size_np
                for batch in range(self.batch_size_np):
                    self.stage_loss_batch_hmindex = [0] * self.num_joints
                    for hmindex in range(self.num_joints):
                        self.stage_loss_batch_hmindex[hmindex] = tf.nn.l2_loss(
                            self.stage_heatmap[stage][batch, :, :, hmindex] -
                            self.gt_heatmap[batch, :, :, hmindex]) * self.train_weights[batch][hmindex]
                    self.stage_loss_batch[batch] = tf.reduce_sum(self.stage_loss_batch_hmindex)
                self.stage_loss[stage] = tf.reduce_sum(self.stage_loss_batch) / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        # 计算总loss
        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            step = tf.cast(self.global_step, dtype=tf.float64)
            # new learning rate setting method
            epcho = step * self.batch_size_np / self.total_num
            new_lr = tf.cond(tf.less(epcho, tf.constant(10.0, dtype=tf.float64)),
                             lambda: 0.0002 / 10.0 * epcho,
                             lambda: tf.cond(tf.less(epcho, tf.constant(20.0, dtype=tf.float64)),
                                     lambda: tf.constant(0.0002, dtype=tf.float64),
                                     lambda: tf.cond(tf.less(epcho, tf.constant(30.0, dtype=tf.float64)),
                                             lambda: 0.0002 - ((epcho - 20.0) / (30.0 - 20.0) * 0.00018),
                                             lambda: tf.cond(tf.less(epcho, tf.constant(35.0, dtype=tf.float64)),
                                                 lambda: tf.constant(0.00002, dtype=tf.float64),
                                                 lambda: tf.cond(tf.less(epcho, tf.constant(40.0, dtype=tf.float64)),
                                                         lambda: tf.constant(0.00001, dtype=tf.float64),
                                                         lambda: 0.00001 - ((epcho - 40.0) / (45.0 - 40.0) * 0.00001)
                             )))))
            self.lr = new_lr
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=self.optimizer)
        self.merged_summary = tf.summary.merge_all()

    def load_weights_from_file(self, weight_file_path, sess, finetune=True):
        weights = pickle.load(open(weight_file_path, 'rb'), encoding='latin1')

        with tf.variable_scope('', reuse=True):
            ## Pre stage conv
            # conv1
            '''
            Pre stage conv
            conv1
            '''
            # for layer,block in enumerate(self.blocks):
            for i in range(1, 3):
                for j in range(2,3):
                    conv_kernel = tf.get_variable('sub_stages' + '/sub_unit_' +str(i) + '/conv' +str(j)+'/kernel')
                    conv_bias = tf.get_variable('sub_stages' + '/sub_unit_' +str()  + '/conv' +str(j)+ '/bias')

                    loaded_kernel = weights['sub_stages' + '/sub_unit_' +str(i) + '/conv' +str(j)]
                    loaded_bias = weights['sub_stages' + '/sub_unit_' +str(i) + '/conv' +str(j)+ '_b']

                    sess.run(tf.assign(conv_kernel, loaded_kernel))
                    sess.run(tf.assign(conv_bias, loaded_bias))
            '''conv2'''
            for i in range(1, 3):
                for j in range(2,3):
                    conv_kernel = tf.get_variable('sub_stages' + '/sub_unit_' +str(i+2) + '/conv' +str(j)+'/kernel')
                    conv_bias = tf.get_variable('sub_stages' + '/sub_unit_' +str(i+2)  + '/conv' +str(j)+ '/bias')

                    loaded_kernel = weights['sub_stages' + '/sub_unit_' +str(i+2) + '/conv' +str(j)]
                    loaded_bias = weights['sub_stages' + '/sub_unit_' +str(i+2) + '/conv' +str(j)+ '_b']

                    sess.run(tf.assign(conv_kernel, loaded_kernel))
                    sess.run(tf.assign(conv_bias, loaded_bias))
            '''conv3'''
            for i in range(1, 5):
                for j in range(2,3):
                    conv_kernel = tf.get_variable('sub_stages' + '/sub_unit_' +str(i+4) + '/conv' +str(j)+'/kernel')
                    conv_bias = tf.get_variable('sub_stages' + '/sub_unit_' +str(i+4)  + '/conv' +str(j)+ '/bias')

                    loaded_kernel = weights['sub_stages' + '/sub_unit_' +str(i+4) + '/conv' +str(j)]
                    loaded_bias = weights['sub_stages' + '/sub_unit_' +str(i+4) + '/conv' +str(j)+ '_b']

                    sess.run(tf.assign(conv_kernel, loaded_kernel))
                    sess.run(tf.assign(conv_bias, loaded_bias))
            '''conv4'''
            for i in range(1, 3):
                for j in range(2,3):
                    conv_kernel = tf.get_variable('sub_stages' + '/sub_unit_' +str(i+8) + '/conv' +str(j)+'/kernel')
                    conv_bias = tf.get_variable('sub_stages' + '/sub_unit_' +str(i+8)  + '/conv' +str(j)+ '/bias')

                    loaded_kernel = weights['sub_stages' + '/sub_unit_' +str(i+8) + '/conv' +str(j)]
                    loaded_bias = weights['sub_stages' + '/sub_unit_' +str(i+8) + '/conv' +str(j)+ '_b']

                    sess.run(tf.assign(conv_kernel, loaded_kernel))
                    sess.run(tf.assign(conv_bias, loaded_bias))
            '''conv4_CPM'''
            for i in range(1, 3):
                for j in range(2,3):
                    conv_kernel = tf.get_variable('sub_stages' + '/sub_unit_' +str(i+10) + '/conv' +str(j)+'/kernel')
                    conv_bias = tf.get_variable('sub_stages' + '/sub_unit_' +str(i+10)  + '/conv' +str(j)+ '/bias')

                    loaded_kernel = weights['sub_stages' + '/sub_unit_' +str(i+10) + '/conv' +str(j)]
                    loaded_bias = weights['sub_stages' + '/sub_unit_' +str(i+10) + '/conv' +str(j)+ '_b']

                    sess.run(tf.assign(conv_kernel, loaded_kernel))
                    sess.run(tf.assign(conv_bias, loaded_bias))
            '''conv5'''
            conv_kernel = tf.get_variable('sub_stages/sub_unit_' + str(15) + '/kernel')
            conv_bias = tf.get_variable('sub_stages/sub_unit_' + str(15) + '/bias')

            loaded_kernel = weights['conv5' + '_CPM']
            loaded_bias = weights['conv5'  + '_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            for i in range(3):
                conv_kernel = tf.get_variable('sub_stages'+'c{}_out'.format(i+1) + '/kernel')
                conv_bias = tf.get_variable('sub_stages'+'c{}_out'.format(i+1) + '/bias')

                loaded_kernel = weights['sub_stages'+'c{}_out'.format(i+1)]
                loaded_bias = weights['sub_stages'+'c{}_out'.format(i+1)+ '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))


            # conv5_3_CPM
            conv_kernel = tf.get_variable('sub_stages/sub_stage_img_feature/kernel')
            conv_bias = tf.get_variable('sub_stages/sub_stage_img_feature/bias')

            loaded_kernel = weights['sub_stages/sub_stage_img_feature']
            loaded_bias = weights['sub_stages/sub_stage_img_feature_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            ## stage 1
            conv_kernel = tf.get_variable('stage_1/conv1/kernel')
            conv_bias = tf.get_variable('stage_1/conv1/bias')

            loaded_kernel = weights['conv5_1_CPM']
            loaded_bias = weights['conv5_1_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            if finetune != True:
                conv_kernel = tf.get_variable('stage_1/stage_heatmap/kernel')
                conv_bias = tf.get_variable('stage_1/stage_heatmap/bias')

                loaded_kernel = weights['conv5_2_CPM']
                loaded_bias = weights['conv5_2_CPM_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))


                ## stage 2 and behind
                for stage in range(2, self.stages + 1):
                    for layer in range(1, 8):
                        conv_kernel = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/kernel')
                        conv_bias = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/bias')

                        loaded_kernel = weights['Mconv' + str(layer) + '_stage' + str(stage)]
                        loaded_bias = weights['Mconv' + str(layer) + '_stage' + str(stage) + '_b']

                        sess.run(tf.assign(conv_kernel, loaded_kernel))
                        sess.run(tf.assign(conv_bias, loaded_bias))