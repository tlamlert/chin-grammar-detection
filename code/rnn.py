import tensorflow as tf
import numpy as np
import sklearn
from tensorflow.keras import Model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from preprocess import get_data
from window_fn import window_data

class RNN(tf.keras.Model):
    def __init__(self, vocab_size, pos_size):

        super(RNN, self).__init__()

        """
        The Model class detects and classifies grammatical errors in a chinese sentence.
        :param
        """
        # hyperparameter
        self.vocab_size = vocab_size  #use for embedding lookup...gives a 3d matrix
        self.pos_size = pos_size
        self.num_classes = 5

        self.embedding_size = 256  #weights
        self.rnn_size = 256
        self.dense_size = 128

        self.batch_size = 100  #number of rows
        self.window_size = 20  #length of sentence/row

        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # model architecture
        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1)) 
        self.pos_E = tf.Variable(tf.random.normal([self.pos_size, self.embedding_size], stddev=.1)) 
        self.RNN = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.classification_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.num_classes),
        ])

    def call(self, inputs, pos):
        """
        :param inputs: a tensor of shape [batch_size, window_size]
        param pos: a tensor of shape [batch_size, window_size]
        :return: probs: a tensor of shape [batch_size, num_classes, rnn_size], and two tensors 
        of shape [batch_size, rnn_size] corresponding to the last output and the cell state 
        """
        # TODO: fill this function
        embeddings = tf.nn.embedding_lookup(self.E, inputs)   #returns tensor with dimensions [bsz, window_sz, embedding_sz]
        pos_embeddings = tf.nn.embedding_lookup(self.pos_E, pos) #returns tensor with dimensions [bsz, window_sz, embedding_sz]
        concat_embeddings = tf.concat([embeddings, pos_embeddings], 2)
        probs1, last_output, cell_state = self.RNN(concat_embeddings) #returns tensor with dimensions [bsz, window_sz, rnn_sz], 2 tensors of [bsz, rnn_sz]
        layer2 = self.classification_layer(probs1) #returns tensor with dimensions [bsz, window_sz, num_classes]
        probs = tf.nn.softmax(layer2)
        return probs, (last_output, cell_state)
        

    def loss_function(self, probs, labels):
        """
        :param probs: a tensor of shape [batch_size, window_size, num_class=5]
        :param labels: a tensor of shape [batch_size, window_size]
        :return: loss: a tensor of shape [1]
        """
        prediction = tf.math.argmax(probs, axis=2)

        mask = tf.not_equal(labels, 0) | tf.not_equal(prediction, 0)
        # print(probs)
        # print(prediction)
        # print(labels)
        # print(mask)

        losses = tf.keras.metrics.sparse_categorical_crossentropy(labels, probs)
        loss = tf.reduce_sum(tf.boolean_mask(losses, mask))
        return loss

        #return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))

    def precision_and_recall(self, prbs, labels):
        """
        Computes the batch precision, recall, and f1 score
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x num_classes]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""
        predictions = tf.argmax(input=prbs, axis=2) #returns [batch_size x window_size]
        print("Predictions", predictions[0])
        print("Labels", labels[0])
        precision = [0]
        recall = [0]
        f1score = [0]
        for i in range(labels.shape[0]):
            precision += sklearn.metrics.precision_score(labels[i], predictions[i], average='macro', zero_division=0)
            recall += sklearn.metrics.recall_score(labels[i], predictions[i], average='macro', zero_division=0)
            f1score += sklearn.metrics.f1_score(labels[i], predictions[i], average='macro', zero_division=0)
            #print(classification_report(labels[i], predictions[i], zero_division=0))
        precision /= labels.shape[0]
        recall /= labels.shape[0]
        f1score /= labels.shape[0]
        #print(precision, recall, f1score)
        return precision, recall, f1score
        

def train(model, train_inputs, train_labels, train_pos):
    num_batches = (int)(train_inputs.shape[0]/model.batch_size)
    
    for i in range(num_batches): 
        inputs_batch = train_inputs[model.batch_size*i:((i+1)*model.batch_size), :]
        labels_batch = train_labels[model.batch_size*i:((i+1)*model.batch_size), :]
        pos_batch = train_pos[model.batch_size*i:((i+1)*model.batch_size), :]

        #backprop
        with tf.GradientTape() as tape: 
            probs, new_state = model.call(inputs_batch, pos_batch)
            loss = model.loss_function(probs, labels_batch)
            print(loss)
            
        #gradient descent
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return

def test(model, test_inputs, test_labels, test_pos):
    num_batches = (int)(test_inputs.shape[0]/model.batch_size)
    loss_list = []
    precision_list = []
    recall_list = []
    f1score_list = []
    #run the model on the test data for all batches 
    for i in range(num_batches): 
        inputs_batch = test_inputs[model.batch_size*i:((i+1)*model.batch_size), :]
        labels_batch = test_labels[model.batch_size*i:((i+1)*model.batch_size), :]
        pos_batch = test_pos[model.batch_size*i:((i+1)*model.batch_size), :]

        probs, new_state = model.call(inputs_batch, pos_batch) 
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
    # file location
    directory = 'processed_dataset/training/npltea16_HSK_TrainingSet/'    
    input_sentence_file = directory + 'input_sentences'
    input_pos_file = directory + 'input_pos'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    train_input, train_input_pos, corrections, train_labels = get_data(input_sentence_file, input_pos_file, correct_sentence_file, label_file)
    
    # create dictionary
    num_inputs = len(train_input) #Num inputs 302494
    # ignore what doesn't fit into the window size
    inputs = train_input[:num_inputs - num_inputs % 300000]  #window_sz = 20 
    labels = train_labels[:num_inputs - num_inputs % 300000]
    pos = train_input_pos[:num_inputs - num_inputs % 300000]

    vocab_list = list(set(inputs))  
    pos_list = list(set(pos))                         
    vocab_dict = dict(zip(vocab_list, range(len(vocab_list))))

    # TODO: read in and tokenize training data
    input_ids = [vocab_dict[x] for x in inputs] 
    
    window_sz = 20
    inputs = window_data(input_ids, window_sz)
    pos = window_data(pos, window_sz)
    labels = window_data(labels, window_sz)

    train_input = inputs[0:12000]
    train_labels = labels[0:12000]
    train_pos = pos[0:12000]

    test_input = inputs[12000:]
    test_labels = labels[12000:]
    test_pos = pos[12000:]
    #corrections = window_data(corrections, window_sz)

    model = RNN(len(vocab_dict), 61)  #pos_list has nums 0-60 but 49 and 59 are missing
    #for i in range(10):
    train(model, train_input, train_labels, train_pos)
    perplexity, precision, recall, f1score = test(model, test_input, test_labels, test_pos)
    print("Perplexity: ", perplexity)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1score: ", f1score)
    


    '''
    Print statements:
    print("Train input", train_input.shape)
    print("Train input pos", train_pos.shape)
    print("Train labels", train_labels.shape)
    print("Corrections", corrections.shape)
    print("Train input", train_input[0])
    print("Train input pos", train_pos[0])
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
