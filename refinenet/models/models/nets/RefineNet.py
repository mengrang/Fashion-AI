import tensorflow as tf
import pickle


class RefineNet(object):
    def __init__(self, total_num, input_size, heatmap_size, batch_size, num_joints, img_type='RGB', is_training=True):    # is_trainng not be used
        self.refine_loss = 0

        self.input_images = None
        self.center_map = None
        self.gt_heatmap = None
        self.predicted_heatmap = None
        self.input_images_resized = None

        self.learning_rate = 0
        self.merged_summary = None

        self.num_joints = num_joints
        self.inference_type = 'Train'  # not be used
        self.batch_size_np = batch_size
        self.stage_loss_batch_hmindex = [0] * self.num_joints
        self.stage_loss_batch = [0] * self.batch_size_np

        self.total_num = total_num

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
                                                  shape=(None,num_joints),
                                                  name='train_weights_placeholder')

        self.predicted_heatmap_placeholder = tf.placeholder(dtype=tf.float32,
                                                            shape=(None, input_size, input_size, num_joints),     # not the same with cpm_hand，hand里头加了1，为何？背景？
                                                            name='gt_hmap_placeholder')

        self.input_images_resized_placeholder = tf.placeholder(dtype=tf.float32,
                                                               shape=(None, heatmap_size, heatmap_size, 3),
                                                               name='input_images_resized_placeholder')
        self._build_model()

    def _build_model(self):
        self._refine()

    def _refine(self):
        self.predicted_heatmap = self.predicted_heatmap_placeholder

        with tf.variable_scope('refine_net'):
            self.input = tf.concat([self.input_images,
                                    self.predicted_heatmap,
                                    ], axis=3)

            sub_conv1 = tf.layers.conv2d(inputs=self.input,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv1')

            sub_conv2 = tf.layers.conv2d(inputs=sub_conv1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv2')

            sub_pool1 = tf.layers.max_pooling2d(inputs=sub_conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',  # not the same with cpm_hand，padding方法不一样
                                                name='sub_pool1')

            sub_conv3 = tf.layers.conv2d(inputs=sub_pool1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv3')

            sub_conv4 = tf.layers.conv2d(inputs=sub_conv3,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv4')

            sub_pool2 = tf.layers.max_pooling2d(inputs=sub_conv4,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',  # not the same with cpm_hand，padding方法不一样
                                                name='sub_pool2')

            sub_conv5 = tf.layers.conv2d(inputs=sub_pool2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv5')

            sub_conv6 = tf.layers.conv2d(inputs=sub_conv5,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv6')

            sub_conv7 = tf.layers.conv2d(inputs=sub_conv6,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv7')

            sub_conv8 = tf.layers.conv2d(inputs=sub_conv7,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv8')

            sub_pool3 = tf.layers.max_pooling2d(inputs=sub_conv8,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',  # not the same with cpm_hand，padding方法不一样
                                                name='sub_pool3')

            # cmp_hand是6个都是512，cpm_body是2个512和4个256
            sub_conv9 = tf.layers.conv2d(inputs=sub_pool3,
                                         filters=512,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv9')

            sub_conv10 = tf.layers.conv2d(inputs=sub_conv9,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv10')

            sub_conv11 = tf.layers.conv2d(inputs=sub_conv10,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv11')

            sub_conv12 = tf.layers.conv2d(inputs=sub_conv11,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv12')

            sub_conv13 = tf.layers.conv2d(inputs=sub_conv12,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv13')

            sub_conv14 = tf.layers.conv2d(inputs=sub_conv13,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv14')

            self.refined_feature = tf.layers.conv2d(inputs=sub_conv14,
                                                    filters=128,
                                                    kernel_size=[3, 3],
                                                    strides=[1, 1],
                                                    padding='same',
                                                    activation=tf.nn.relu,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='refined_feature')

            conv1 = tf.layers.conv2d(inputs=self.refined_feature,
                                     filters=512,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')

            self.refined_heatmap_stage1 = tf.layers.conv2d(inputs=conv1,
                                                           filters=self.num_joints,
                                                           # not the same with cpm_hand，hand里头加了1，为何？背景？
                                                           kernel_size=[1, 1],
                                                           strides=[1, 1],
                                                           padding='same',  # not the same with cpm_hand，padding方法不一样
                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                           name='stage1_heatmap')

            self.current_featuremap = tf.concat([self.refined_heatmap_stage1,
                                                 self.refined_feature,
                                                 ],
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

            self.refined_heatmap = tf.layers.conv2d(inputs=mid_conv6,
                                                    filters=self.num_joints,  # not the same with cpm_hand，hand里头加了1，为何？背景？
                                                    kernel_size=[1, 1],
                                                    strides=[1, 1],
                                                    padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='mid_conv7')

    # new learning rate setting method
    def build_loss3(self, optimizer='Adam'):
        self.gt_heatmap = self.gt_hmap_placeholder
        self.train_weights = self.train_weights_placeholder
        self.refine_loss = 0
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

        # 计算每个stage的loss, weighted l2 loss
        with tf.variable_scope('refine_loss'):
            self.stage_loss_batch = [0] * self.batch_size_np
            for batch in range(self.batch_size_np):
                self.stage_loss_batch_hmindex = [0] * self.num_joints
                for hmindex in range(self.num_joints):
                    self.stage_loss_batch_hmindex[hmindex] = tf.nn.l2_loss(self.refined_heatmap[batch,:,:,hmindex] -
                                                                           self.gt_heatmap[batch,:,:,hmindex]) * \
                                                                           self.train_weights[batch][hmindex]
                self.stage_loss_batch[batch] = tf.reduce_sum(self.stage_loss_batch_hmindex)

            self.refine_loss = tf.reduce_sum(self.stage_loss_batch) / self.batch_size

        tf.summary.scalar('refine_loss', self.refine_loss)

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

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.refine_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=self.optimizer)
        self.merged_summary = tf.summary.merge_all()
