import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BasicFeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3):
        print("Initialising Feature network")
        self.state_size = state_size
        self.lr = learning_rate
        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # shared convolutional layers
        conv1 = layers.Conv2D(16, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(bn1)
        bn2 = layers.BatchNormalization()(conv2)
        conv3 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(bn2)
        bn3 = layers.BatchNormalization()(conv3)
        f1 = layers.Flatten()(bn3)
        fc1 = layers.Dense(128, activation='relu')(f1)
        fc2 = layers.Dense(64, activation='relu')(fc1)
        model = tf.keras.Model(inputs=img_input, outputs=fc2)
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)


class FeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3):
        print("Initialising Feature network")
        self.state_size = state_size
        self.lr = learning_rate
        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # CNN
        # Shared convolutional layers
        conv1 = layers.Conv2D(16, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(bn1)
        conv3 = layers.Conv2D(64, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(conv2)
        mp1 = layers.MaxPool2D(padding="SAME")(conv3)
        bn2 = layers.BatchNormalization()(mp1)
        conv4 = layers.Conv2D(128, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(bn2)
        conv5 = layers.Conv2D(128, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(conv4)
        mp2 = layers.MaxPool2D(padding="SAME")(conv5)
        bn3 = layers.BatchNormalization()(mp2)
        f1 = layers.Flatten()(bn3)

        fc1 = layers.Dense(128, activation='relu')(f1)
        fc2 = layers.Dense(64, activation='relu')(fc1)
        model = tf.keras.Model(inputs=img_input, outputs=fc2)
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)


class AttentionFeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3):
        print("Initialising Feature network")
        self.state_size = state_size
        self.lr = learning_rate
        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # CNN
        # Shared convolutional layers
        conv1 = layers.Conv2D(16, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(bn1)
        conv3 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(conv2)
        mp1 = layers.MaxPool2D(padding="SAME")(conv3)
        bn2 = layers.BatchNormalization()(mp1)
        conv4 = layers.Conv2D(64, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(bn2)
        conv5 = layers.Conv2D(64, kernel_size=5, strides=2,
                              padding="SAME", activation='relu')(conv4)
        mp2 = layers.MaxPool2D(padding="SAME")(conv5)
        bn3 = layers.BatchNormalization()(mp2)
        f1 = layers.Flatten()(bn3)

        # Attention
        q = layers.Reshape((4, 16))(f1)
        att = layers.Attention()([q, q])
        # att = layers.MultiHeadAttention(num_heads=3, key_dim=2)([q, q])

        # Output
        f2 = layers.Reshape((1, 64))(att)
        f2 = layers.Flatten()(f2)
        fc1 = layers.Dense(128, activation='relu')(f2)
        fc2 = layers.Dense(64, activation='relu')(fc1)
        model = tf.keras.Model(inputs=[img_input], outputs=fc2)
        print('shared feature attention network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)

