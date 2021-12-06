import tensorflow as tf
import numpy as np
from preprocess import get_data
from window_fn import window_data

class RNN(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class detects and classifies grammatical errors in a chinese sentence.
        :param
        """
        # hyperparameter
        self.vocab_size = vocab_size  #use for embedding lookup...gives a 3d matrix
        self.num_classes = 5

        self.embedding_size = 256  #weights
        self.rnn_size = 256
        self.dense_size = 30 #128

        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.batch_size = 100  #number of rows
        self.window_size = 20  #length of sentence/row

        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # model architecture
        #self.embedding_dict, self.embedding_size = load_embedding()
        # TODO: check this rnn model
        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1)) 
        self.RNN = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.classification_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.num_classes),
        ])

    def call(self, inputs):
        """

        :param inputs: a tensor of shape [batch_size, window_size]
        :return: probs: a tensor of shape [batch_size, num_classes, rnn_size], and two tensors 
        of shape [batch_size, rnn_size] corresponding to the last output and the cell state 
        """
        # TODO: fill this function
        embeddings = tf.nn.embedding_lookup(self.E, inputs)   #returns tensor with dimensions [bsz, window_sz, embedding_sz]
        probs1, last_output, cell_state = self.RNN(embeddings) #returns tensor with dimensions [bsz, window_sz, rnn_sz], 2 tensors of [bsz, rnn_sz]
        layer2 = self.classification_layer(probs1) #returns tensor with dimensions [bsz, window_sz, num_classes]
        probs = tf.nn.softmax(layer2)
        return probs, (last_output, cell_state)
        '''
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
        '''

    def loss_function(self, probs, labels):
        """

        :param probs: a tensor of shape [batch_size, window_size, num_class=5]
        :param labels: a tensor of shape [batch_size, window_size]
        :return: loss: a tensor of shape [1]
        """
        # loss function: -y * log(n) - (1 - y) log(1-n)
        # Where n(.) represent the probability in the training corpus
        '''
        np.argmax
        one hot vector
        element-wise multiplication

        loss = tf.reduce_sum()
        return loss
        '''

        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))

def train(model, train_inputs, train_labels):
    num_batches = (int)(train_inputs.shape[0]/model.batch_size)
    
    for i in range(num_batches): 
        inputs_batch = train_inputs[model.batch_size*i:((i+1)*model.batch_size), :]
        labels_batch = train_labels[model.batch_size*i:((i+1)*model.batch_size), :]
    
        #backprop
        with tf.GradientTape() as tape: 
            probs, new_state = model.call(inputs_batch, None)
            loss = model.loss(probs, labels_batch)
            
        #gradient descent
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return

def test(model, test_inputs, test_labels):
    num_batches = (int)(test_inputs.shape[0]/model.batch_size)
    loss_list = []
    #run the model on the test data for all batches 
    for i in range(num_batches): 
        inputs_batch = test_inputs[model.batch_size*i:((i+1)*model.batch_size), :]
        labels_batch = test_labels[model.batch_size*i:((i+1)*model.batch_size), :]
        probs, new_state = model.call(inputs_batch, None) 
        loss_list.append(model.loss(probs, labels_batch))

    perplexity = np.exp(tf.reduce_mean(loss_list))  
    
    return perplexity


def load_embedding():
    token_list = np.load("data/embedding_word/token_list.npy")
    vector_list = np.load("data/embedding_word/vector_list.npy")
    dictionary = dict(zip(token_list, vector_list))
    dimension = vector_list.shape[1]
    return dictionary, dimension


def main():
    vocab_dict, dimension = load_embedding()  #size of embedding 300

    # file location
    directory = '../processed_dataset/training/npltea16_HSK_TrainingSet/'
    input_sentence_file = directory + 'input_sentences'
    input_pos_file = directory + 'input_pos'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    train_input, train_input_pos, _, train_labels = get_data(input_sentence_file, input_pos_file, correct_sentence_file, label_file)
    window_sz = 20
    train_input = window_data(train_input, window_sz)
    train_input_pos = window_data(train_input_pos, window_sz)
    train_labels = window_data(train_labels, window_sz)

    model = RNN(len(vocab_dict))
    train(model, train_input, train_labels)
    perplexity, accuracy = test(model, test_input, test_labels)
    print("Perplexity: ", perplexity)
    print("Accuracy: ", accuracy)

    #train_ids [number sentences, sentence length] these are the word embeddings corresponding to dictionary
    #train_grammar_labels [number sentences, sentence length] these are the grammar embeddings 0-4 for type of error
    #test_ids
    #test_labels


    #Inputs:
    # num batches * window size
    #words
    #dictionary
    
    #Labels:
    #num batches * window size
    #numbers 0-4
