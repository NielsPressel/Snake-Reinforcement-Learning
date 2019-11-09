import tensorflow as tf


class DeepQNetwork(tf.keras.Model):

    def __init__(self, width, height, frame_count, action_size):
        super(DeepQNetwork, self).__init__()
        self.__conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(width, height, frame_count))
        self.__max_pool_1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.__conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.__max_pool_2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.__flat = tf.keras.layers.Flatten()
        self.__dense_1 = tf.keras.layers.Dense(20, activation="relu")
        self.__dense_2 = tf.keras.layers.Dense(action_size, activation="linear")

    def call(self, x):
        x = self.__conv_1(x)
        x = self.__max_pool_1(x)
        x = self.__conv_2(x)
        x = self.__max_pool_2(x)
        x = self.__flat(x)
        x = self.__dense_1(x)
        return self.__dense_2(x)
