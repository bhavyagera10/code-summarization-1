import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True))

from groundhog.trainer import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation, Flatten, Dense, Dropout, RepeatVector
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.engine.topology import Merge
from keras.preprocessing import sequence
import time
from keras import callbacks
from skimage.feature.tests.test_util import plt
import keras

import data_generator


max_caption_len = 20
maxlen = 600

data_gen = data_generator.DataGenerator("../data/code_f_keyword_indexed.txt", "../data/comment_f_keyword_indexed.txt", 0.20,
                                        maxlen, max_caption_len)

codes, next_words = data_gen.MakeDataset(train=True)
#
codes = sequence.pad_sequences(codes, maxlen=maxlen)


codesT, next_wordsT = data_gen.MakeDataset(train=False)
codesT = sequence.pad_sequences(codesT, maxlen=maxlen)


vocab_size = 4000

max_features = 6000  # size of token

nb_output = data_gen.vocab_size  # len(Y_train[0])

nb_filter = 128
filter_length = 10
hidden_dims = 256
embedding_dims = 128

code_model = Sequential()
code_model.add(Embedding(max_features, embedding_dims, input_length=maxlen, dropout=0.2))

code_model.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=10,
                             border_mode='valid',
                             subsample_length=1))
code_model.add(Activation('relu'))

code_model.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=5,
                             border_mode='valid',
                             subsample_length=1))
code_model.add(Activation('relu'))

code_model.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=3,
                             border_mode='valid',
                             subsample_length=1))
code_model.add(Activation('relu'))
code_model.add(MaxPooling1D(pool_length=code_model.output_shape[1]))
code_model.add(Flatten())
# We add a vanilla hidden layer:
code_model.add(Dense(256))
code_model.add(Activation('relu'))

code_model.add(Dropout(0.5))
code_model.add(Dense(nb_output))
code_model.add(Activation('sigmoid'))

code_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

print "train"
early = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
cp = keras.callbacks.ModelCheckpoint('../model/keyword_f.h5', monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=True, mode='auto')

del data_gen

code_model.fit(codes, next_words, batch_size=32, nb_epoch=500, validation_data=(codesT, next_wordsT), callbacks=[early])

json_string = code_model.to_json()
open('../model/keyword_f.json', 'w').write(json_string)
code_model.save_weights('../model/keyword_f.h5')
