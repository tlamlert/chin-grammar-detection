import tensorflow as tf
import numpy as np

class RNN(tf.keras.Model):
    def __init__(self, num_class=5):
        """
        The Model class detects and classifies grammatical errors in a chinese sentence.
        :param
        """
        # hyperparameter
        self.batch_size = 100
        self.window_size = 20
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # model architecture
        self.embedding_dict, self.embedding_size = load_embedding()
        # TODO: check this rnn model
        self.RNN = tf.keras.layers.SimpleRNN(30, return_sequences=True)
        self.classification_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(num_class),
        ])

    def call(self, inputs):
        """

        :param inputs: a tensorflow of shape [batch_size, window_size]
        :return:
        """
        # TODO: fill this function
        # embedding lookup
        embedded = []
        for idx, sentence in enumerate(inputs):
            embedded.append([self.embedding_dict[x] for x in sentence])
        embedded = tf.convert_to_tensor(embedded)
        # embedded: a tensor of shape [batch_size, window_size, embedding_size]

        # RNN
        hidden_states = self.RNN(embedded)
        # hidden_states: a tensor of shape [batch_size, window_size]

        # classification layer
        error_prop = self.classification_layer(hidden_states)
        # error_prop: a tensor of shape [batch_size, window_size, num_class=5]

        return error_prop

    def loss_function(self, probs, labels):
        """

        :param probs: a tensor of shape [batch_size, window_size, num_class=5]
        :param labels: a tensor of shape [batch_size, window_size]
        :return: loss: a tensor of shape [1]
        """
        # loss function: -y * log(n) - (1 - y) log(1-n)
        # Where n(.) represent the probability in the training corpus
        np.argmax
        one hot vector
        element-wise multiplication

        loss = tf.reduce_sum()
        return loss

def train(model, inputs, labels):

    pass

def test(model, inputs, labels):

    return perplexity, accuracy

def load_embedding():
    token_list = np.load("data/embedding_word/token_list.npy")
    vector_list = np.load("data/embedding_word/vector_list.npy")
    dict = dict(zip(token_list, vector_list))
    dimension = vector_list.shape[1]
    return dict, dimension