import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data
from window_fn import window_data

class RNN(tf.keras.Model):
    def __init__(self, vocab_size):

        super(RNN, self).__init__()

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

    def precision_and_recall(self, prbs, labels):
        """
        Computes the batch precision, recall, and f1 score
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x num_classes]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""
        #print("Probs shape", prbs.shape)
        #print("Probs", prbs[0])
        #print("Labels shape", labels.shape)
        #print("Labels", labels[0])
        predictions = tf.argmax(input=prbs, axis=2) #returns [batch_size x window_size]
        #print("Predictions shape", predictions.shape)
        #print("Predictions", predictions[0])
        cm = np.zeros((self.num_classes, self.num_classes))
        true_pos = np.zeros(self.num_classes)
        for i in range(labels.shape[0]):
            cm += tf.math.confusion_matrix(labels[i], predictions[i], 5).numpy()
            true_pos += np.diag(cm)
        print("Confusion matrix", cm)
        print("True pos", true_pos)
        print("np.sum axis 0", np.sum(cm, axis=0))
        print("np.sum axis 1", np.sum(cm, axis=1))
        precision = np.mean(np.divide(true_pos, np.sum(cm, axis=0)))
        print("Precision", precision)
        recall = np.mean(np.divide(true_pos, np.sum(cm, axis=1)))
        print("Recall", recall)
        f1score = 2* ((precision * recall)/(precision + recall))
        print("F1score", f1score)
        return precision, recall, f1score


def train(model, train_inputs, train_labels):
    num_batches = (int)(train_inputs.shape[0]/model.batch_size)
    
    for i in range(num_batches): 
        inputs_batch = train_inputs[model.batch_size*i:((i+1)*model.batch_size), :]
        labels_batch = train_labels[model.batch_size*i:((i+1)*model.batch_size), :]
    
        #backprop
        with tf.GradientTape() as tape: 
            probs, new_state = model.call(inputs_batch)
            loss = model.loss_function(probs, labels_batch)
            print(loss)
            
        #gradient descent
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return

def test(model, test_inputs, test_labels):
    num_batches = (int)(test_inputs.shape[0]/model.batch_size)
    loss_list = []
    precision_list = []
    recall_list = []
    f1score_list = []
    #run the model on the test data for all batches 
    for i in range(num_batches): 
        inputs_batch = test_inputs[model.batch_size*i:((i+1)*model.batch_size), :]
        labels_batch = test_labels[model.batch_size*i:((i+1)*model.batch_size), :]
        probs, new_state = model.call(inputs_batch) 
        loss = model.loss_function(probs, labels_batch)
        loss_list.append(loss)
        print("Test loss", loss)
        precision, recall, f1score = model.precision_and_recall(probs, labels_batch)
        precision_list.append(precision)
        recall_list.append(recall)
        f1score_list.append(f1score)

    perplexity = np.exp(tf.reduce_mean(loss_list))  
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1score = np.mean(f1score_list)

    return perplexity, avg_precision, avg_recall, avg_f1score

'''
def load_embedding():
    token_list = np.load("data/embedding_character/token_list.npy")
    print("Token_list", token_list.shape)
    vector_list = np.load("data/embedding_character/vector_list.npy")
    print("Vector_list", vector_list.shape)
    dictionary = dict(zip(token_list, vector_list))
    dimension = vector_list.shape[1]
    return dictionary, dimension
'''

if __name__ == '__main__':
    ''' 
    DON'T USE THIS
    vocab_dict, dimension = load_embedding()  #size of embedding 300
    print("Vocab dict", len(vocab_dict))
    print("Dimension", dimension)
    
    #Token_list (13136,)
    #Vector_list (13136, 300)
    #Vocab dict 13136
    #Dimension 300
    '''

    # file location
    directory = 'processed_dataset/training/npltea16_HSK_TrainingSet/'    
    input_sentence_file = directory + 'input_sentences'
    input_pos_file = directory + 'input_pos'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    train_input, train_input_pos, corrections, train_labels = get_data(input_sentence_file, input_pos_file, correct_sentence_file, label_file)
    
    # create dictionary
    num_inputs = len(train_input)
    #print("Num inputs", num_inputs)  #Num inputs 302494
    # ignore what doesn't fit into the window size
    inputs = train_input[:num_inputs - num_inputs % 300000]  #window_sz = 20 
    labels = train_labels[:num_inputs - num_inputs % 300000]
    #print("Inputs", len(inputs))  #Inputs 300000
    vocab_list = list(set(inputs))
    vocab_dict = dict(zip(vocab_list, range(len(vocab_list))))
    #print("My dict length", len(vocab_dict))  #My dict length 22699

    
    # TODO: read in and tokenize training data
    input_ids = [vocab_dict[x] for x in inputs] 
    # TODO: read in and tokenize testing data
    #test_ids = [vocab_dict[x] for x in test_list] 
    
    
    window_sz = 20
    inputs = window_data(input_ids, window_sz)
    #train_input_pos = window_data(train_input_pos, window_sz)
    labels = window_data(labels, window_sz)
    train_input = inputs[0:12000]
    train_labels = labels[0:12000]
    test_input = inputs[12000:]
    test_labels = labels[12000:]
    #corrections = window_data(corrections, window_sz)
    
    print("Train input", train_input[0])
    print("Train labels", train_labels[0])
    print("Test input", test_input[1])
    print("Test labels", test_labels[1])

    '''
    print("Here")
    print("Train input", train_input.shape)
    print("Train input pos", train_input_pos.shape)
    print("Train labels", train_labels.shape)
    print("Corrections", corrections.shape)
    print("Train input", train_input[0])
    print("Train input pos", train_input_pos[0])
    print("Train labels", train_labels[0])
    print("Corrections", corrections[0])
    
    Train input (15124, 20)
    Train input pos (22123, 20)
    Train labels (15124, 20)
    Corrections (15167, 20)
    Train input ['别' '只' '能' '想' '自己' '，' '想' '你' '周围' '的' '人' '。' '还' '有' '你' '，' '如果'
    '你' '是' '一']
    Train input pos [38 39  3 15  8 10  2 15 10 16 38  4  1  9 38 32 10  2 14  1]
    Train labels [0 0 1 0 0 0 2 2 0 0 0 0 0 0 1 0 0 0 0 0]
    Corrections ['别' '只' '想' '自己' '，' '要' '想想' '你' '周围' '的' '人' '。' '还' '有' '，' '如果' '你'
    '是' '一' '个']
    '''

    
    model = RNN(len(vocab_dict))
    train(model, train_input, train_labels)
    perplexity, precision, recall, f1score = test(model, test_input, test_labels)
    print("Perplexity: ", perplexity)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1score: ", f1score)
    


    #*************************
    #Comments about inputs
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
