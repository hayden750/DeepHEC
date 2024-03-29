import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from attention import SeqSelfAttention


class BasicFeatureNetwork:
    def __init__(self, state_size, learning_rate=2e-4):
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
        fc1 = layers.Dense(64, activation='relu')(f1)
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
    def __init__(self, state_size, learning_rate=2e-4):
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

        fc1 = layers.Dense(128, activation='relu')(f1)
        fc2 = layers.Dense(128, activation='relu')(fc1)
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
        self.state_size = state_size
        self.lr = learning_rate
        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # shared convolutional layers
        x = layers.Conv2D(16, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)

        # First attention layer
        # Bahdanau
        # x = layers.AdditiveAttention()([x, x])
        # + sig activation and multiply
        # attn1 = layers.Attention()([x, x])
        # attn1 = layers.AdditiveAttention()([x, x])
        # attn1 = tf.keras.activations.sigmoid(attn1)
        # x = layers.Add()([attn1, x])
        # x = layers.Multiply()([attn1, x])

        # Luong-style
        x = layers.Attention()([x, x])

        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(x)

        # Second attention layer
        # x = layers.AdditiveAttention()([x, x])
        # attn2 = layers.AdditiveAttention()([x, x])
        # attn2 = layers.Attention()([x, x])
        # attn2 = tf.keras.activations.sigmoid(attn2)
        # x = layers.Add()([attn2, x])
        # x = layers.Multiply()([attn2, x])

        x = layers.Attention()([x, x])

        # x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(x)
        # bn3 = layers.BatchNormalization()(conv3)

        # Third attention layer
        # x = layers.AdditiveAttention()([x, x])
        # attn3 = layers.AdditiveAttention()([x, x])
        # attn3 = layers.Attention()([x, x])
        # attn3 = tf.keras.activations.sigmoid(attn3)
        # x = layers.Multiply()([attn3, x])
        # x = layers.Add()([attn3, x])

        # x = layers.MultiHeadAttention(num_heads=2, key_dim=36)(x, x)        # does not work well

        x = layers.Attention()([x, x])

        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Flatten()(x)

        # Sequential Attention
        # x = layers.Reshape((8, 32))(x)
        # x = SeqSelfAttention(units=32)(x)
        # x = layers.Reshape((1, 256))(x)
        # x = layers.Flatten()(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        model = tf.keras.Model(inputs=img_input, outputs=x, name='feature_net')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='att_feature_net.pdf',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)
