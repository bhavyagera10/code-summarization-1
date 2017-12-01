'''
Created on 2016. 9. 30.

@author: KSA
'''
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation, Flatten, Dense, Dropout, RepeatVector
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.engine.topology import Merge


import logging
import sys
from keras.regularizers import l2
from keras.engine.training import Model


# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class CGModel(object):
    '''
    classdocs
    '''


    def __init__(self, max_features=20000, code_maxlen, embed_size=100, hidden_size=200, vocab_size, optimiser, weights=None, gru=False,
                 clipnorm=-1, batch_size=None, t=None, lr=0.001):
        
        self.max_t = t  # Expected timesteps. Needed to build the Theano graph

        # Model hyperparameters
        self.max_features = max_features
        self.vocab_size = vocab_size  # size of word vocabulary
        self.embed_size = embed_size  # number of units in a word embedding
        self.hidden_size = hidden_size  # number of units in first LSTM

        # Optimiser hyperparameters
        self.optimiser = optimiser  # optimisation method
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.clipnorm = clipnorm

        self.weights = weights  # initialise with checkpointed weights?
        
        
        


def buildModel(self):
        '''
        Define the exact structure of your model here. We create an image
        description generation model by merging the VGG image features with
        a word embedding model, with an LSTM over the sequences.
        '''
        logger.info('Building Keras model...')

                
        
        code_model = Sequential()
        code_model.add(Embedding(self.max_features, self.embed_size, input_length=maxlen,dropout=0.2))
         
        code_model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                subsample_length=1))
        #model.add(BatchNormalization())
        code_model.add(Activation('relu'))
        
        code_model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                subsample_length=1))
        # model.add(BatchNormalization())
        code_model.add(Activation('relu'))
        
        code_model.add(MaxPooling1D(pool_length=code_model.output_shape[1]))
        code_model.add(Flatten())
        # We add a vanilla hidden layer:
        code_model.add(Dense(128))
        
        
        # next, let's define a RNN model that encodes sequences of words
        # into sequences of 128-dimensional word vectors.
        language_model = Sequential()
        language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
        language_model.add(LSTM(output_dim=128, return_sequences=True))
        language_model.add(TimeDistributed(Dense(128)))
        
        # let's repeat the image vector to turn it into a sequence.
        code_model.add(RepeatVector(max_caption_len))
        
        # the output of both models will be tensors of shape (samples, max_caption_len, 128).
        # let's concatenate these 2 vector sequences.
        model = Sequential()
        model.add(Merge([code_model, language_model], mode='concat', concat_axis=-1))
        # let's encode this vector sequence into a single vector
        model.add(LSTM(256, return_sequences=False))
        # which will be used to compute a probability
        # distribution over what the next word in the caption should be!
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
        
        
        #model.fit([codes, partial_captions], next_words, batch_size=16, nb_epoch=100)
        
        
