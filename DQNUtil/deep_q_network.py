import os

import tensorflow as tf


# This class is written in tensorflow 1.x style, just for reference prurposes


class DeepQNetwork:

    def __init__(self, lr, number_actions, name, fc1_dims=1024, input_dims=(20, 20, 3), checkpoint_dir="tmp/dqn"):
        self.lr = lr
        self.number_actions = number_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.checkpoint_dir = checkpoint_dir
        self.input_dims = input_dims

        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.checkpoint_dir, "deepqnetwork.ckpt")
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            self.actions = tf.placeholder(tf.float32, shape=[None, self.number_actions], name='action_taken')
            self.q_target = tf.placeholder(tf.float32, shape=[None, self.number_actions], name='q_value')

            conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=(4, 4), strides=4, name='conv1',
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv1_activated = tf.nn.relu(conv1)

            flat = tf.layers.flatten(conv1_activated)
            dense1 = tf.layers.dense(flat, units=self.fc1_dims, activation=tf.nn.relu,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            dense2 = tf.layers.dense(dense1, units=512, activation=tf.nn.relu,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))

            self.q_values = tf.layers.dense(dense2, units=self.number_actions,
                                            kernel_initializer=tf.variance_scaling_initializer(scale=2))
            self.loss = tf.reduce_mean(tf.square(self.q_values - self.q_target))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def load_checkpoint(self):
        print("---Loading checkpoint---")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("---Saving checkpoint---")
        self.saver.save(self.sess, self.checkpoint_file)
