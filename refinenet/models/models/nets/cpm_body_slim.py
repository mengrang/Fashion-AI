import pickle
import tensorflow as tf
import tensorflow.contrib.slim as slim
# slim库是tensorflow中的一个高层封装，它将原来很多tf中复杂的函数进一步封装，
# 省去了很多重复的参数，以及平时不会考虑到的参数。可以理解为tensorflow的升级版。


class CPM_Model(object):
    def __init__(self, stages, joints):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 0

    def build_model(self, input_image, center_map, batch_size):
        self.batch_size = batch_size
        self.input_image = input_image
        self.center_map = center_map

        # 暂时不明白center_map是什么
        # center map is a gaussion template which gather the respose
        with tf.variable_scope('pooled_center_map'):
            self.center_map = slim.avg_pool2d(self.center_map,
                                              [9, 9], stride=8,
                                              padding='SAME',
                                              scope='center_map')
        # slim.arg_scope可以定义一些函数的默认参数值，在scope内，我们重复用到这些函数时可以不用把所有参数都写一遍
        # 一个slim.arg_scope内可以用list来同时定义多个函数的默认参数（前提是这些函数都有这些参数），
        # 另外，slim.arg_scope也允许相互嵌套。在其中调用的函数，可以不用重复写一些参数（例如kernel_size=[3, 3]），
        # 但也允许覆盖（例如最后一行，卷积核大小为[5，5]）。
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            # image_feature 提取，论文中的x
            with tf.variable_scope('sub_stages'):
                net = slim.conv2d(input_image, 64, [3, 3], scope='sub_conv1')
                net = slim.conv2d(net, 64, [3, 3], scope='sub_conv2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool1')
                net = slim.conv2d(net, 128, [3, 3], scope='sub_conv3')
                net = slim.conv2d(net, 128, [3, 3], scope='sub_conv4')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool2')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv5')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv6')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv7')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv8')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool3')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv9')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv10')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv11')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv12')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv13')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv14')
                self.sub_stage_img_feature = slim.conv2d(net, 128, [3, 3],
                                                         scope='sub_stage_img_feature')

            # 第一阶段
            with tf.variable_scope('stage_1'):
                conv1 = slim.conv2d(self.sub_stage_img_feature, 512, [1, 1],
                                    scope='conv1')
                # joints是什么？
                # stage_heatmap append一第一阶段的heatmap
                self.stage_heatmap.append(slim.conv2d(conv1, self.joints, [1, 1],
                                                      scope='stage_heatmap'))
            # 后续阶段
            for stage in range(2, self.stages+1):
                self._middle_conv(stage)

    # 中间阶段定义函数
    def _middle_conv(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            # 前一阶段各部件响应→ 空间特征 + 阶段性的卷积结果（46*46*32）→ 纹理特征 + 中心约束
            self.current_featuremap = tf.concat([self.stage_heatmap[stage-2],   # 从stage=2开始，即从stage_heatmap[0]开始
                                                 self.sub_stage_img_feature,    # 加提取出来的图像特征图
                                                 self.center_map],              # 加center_map
                                                axis=3)
            # 拼接之后再进行卷积
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                mid_net = slim.conv2d(self.current_featuremap, 128, [7, 7], scope='mid_conv1')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid_conv2')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid_conv3')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid_conv4')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid_conv5')
                mid_net = slim.conv2d(mid_net, 128, [1, 1], scope='mid_conv6')
                self.current_heatmap = slim.conv2d(mid_net, self.joints, [1, 1],
                                                   scope='mid_conv7')
                # append新的heatmap
                self.stage_heatmap.append(self.current_heatmap)

    # 计算损失，gt_heatmap 是groundtruth heatmap
    def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        # 每一阶段单独计算损失，这就是中继监督优化
        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage+1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            # summary scalar，准备tensorboard可视化
            tf.summary.scalar('stage' + str(stage+1) + '_loss', self.stage_loss[stage])

        # 再计算所有stage的loss，这就是全局优化
        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]

            # summary scalar，准备tensorboard可视化
            tf.summary.scalar('total loss', self.total_loss)

        # 定义训练优化方法
        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_creat_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer='Adam')

        # 一次性merge所有summary的数据
        self.merged_summary = tf.summary.merge_all()

    # 从文件加载权重
    def load_weights_from_file(self, weight_file_path, sess, finetune=True):
        weights = pickle.load(open(weight_file_path, 'rb'), encoding='latin1')

        with tf.variable_scope('', reuse=True):
            ## Pre stage conv
            # for layer in range(1, 15):
            #     conv_weights = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/weights')
            #     conv_biases = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/biases')
            #
            #     loaded_weights = weights['sub_conv' + str(layer)]
            #     loaded_biases = weights['sub_conv' + str(layer)]
            #
            #     sess.run(tf.assign(conv_weights, loaded_weights))
            #     sess.run(tf.assign(conv_biases, loaded_biases))

            # conv1
            for layer in range(1, 3):
                conv_weights = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/weights')
                conv_biases = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/biases')

                loaded_weights = weights['conv1_' + str(layer)]
                loaded_biases = weights['conv1_' + str(layer) + '_b']

                sess.run(tf.assign(conv_weights, loaded_weights))
                sess.run(tf.assign(conv_biases, loaded_biases))

            # conv2
            for layer in range(1, 3):
                conv_weights = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/weights')
                conv_biases = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/biases')

                loaded_weights = weights['conv2_' + str(layer)]
                loaded_biases = weights['conv2_' + str(layer) + '_b']

                sess.run(tf.assign(conv_weights, loaded_weights))
                sess.run(tf.assign(conv_biases, loaded_biases))

            # conv3
            for layer in range(1, 5):
                conv_weights = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/weights')
                conv_biases = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/biases')

                loaded_weights = weights['conv3_' + str(layer)]
                loaded_biases = weights['conv3_' + str(layer) + '_b']

                sess.run(tf.assign(conv_weights, loaded_weights))
                sess.run(tf.assign(conv_biases, loaded_biases))

            # conv4
            for layer in range(1, 3):
                conv_weights = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/weights')
                conv_biases = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/biases')

                loaded_weights = weights['conv4_' + str(layer)]
                loaded_biases = weights['conv4_' + str(layer) + '_b']

                sess.run(tf.assign(conv_weights, loaded_weights))
                sess.run(tf.assign(conv_biases, loaded_biases))

            # conv4_CPM
            for layer in range(1, 5):
                conv_weights = tf.get_variable('sub_stages/sub_conv' + str(layer + 10) + '/weights')
                conv_biases = tf.get_variable('sub_stages/sub_conv' + str(layer + 10) + '/biases')

                loaded_weights = weights['conv4_' + str(2 + layer) + '_CPM']
                loaded_biases = weights['conv4_' + str(2 + layer) + '_CPM_b']

                sess.run(tf.assign(conv_weights, loaded_weights))
                sess.run(tf.assign(conv_biases, loaded_biases))

            # conv5_3_CPM
            conv_weights = tf.get_variable('sub_stages/sub_stage_img_feature/weights')
            conv_biases = tf.get_variable('sub_stages/sub_stage_img_feature/biases')

            loaded_weights = weights['conv4_7_CPM']
            loaded_biases = weights['conv4_7_CPM_b']

            sess.run(tf.assign(conv_weights, loaded_weights))
            sess.run(tf.assign(conv_biases, loaded_biases))

            ## stage 1
            conv_weights = tf.get_variable('stage_1/conv1/weights')
            conv_biases = tf.get_variable('stage_1/conv1/biases')

            loaded_weights = weights['conv5_1_CPM']
            loaded_biases = weights['conv5_1_CPM_b']

            sess.run(tf.assign(conv_weights, loaded_weights))
            sess.run(tf.assign(conv_biases, loaded_biases))

            if finetune != True:
                conv_weights = tf.get_variable('stage_1/stage_heatmap/weights')
                conv_biases = tf.get_variable('stage_1/stage_heatmap/biases')

                loaded_weights = weights['conv5_2_CPM']
                loaded_biases = weights['conv5_2_CPM_b']

                sess.run(tf.assign(conv_weights, loaded_weights))
                sess.run(tf.assign(conv_biases, loaded_biases))

                ## stage 2 and behind
                for stage in range(2, self.stages + 1):
                    for layer in range(1, 8):
                        conv_weights = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/weights')
                        conv_biases = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/biases')

                        loaded_weights = weights['Mconv' + str(layer) + '_stage' + str(stage)]
                        loaded_biases = weights['Mconv' + str(layer) + '_stage' + str(stage) + '_b']

                        sess.run(tf.assign(conv_weights, loaded_weights))
                        sess.run(tf.assign(conv_biases, loaded_biases))
