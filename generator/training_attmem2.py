
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True))

from keras.models import Sequential
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation, Flatten, Dense, Dropout, RepeatVector
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import RMSprop
from keras.layers import Input, LSTM, Embedding, Dense

import data_generator
from keras.preprocessing import sequence
import keras

from keras.models import Model

from bleu import computeMaps, bleuFromMaps
from text_generator import SentenceGeneration

max_caption_len = 26
maxlen = 500
mem_size = 30

data_gen = data_generator.DataGenerator("../qnaData_all/code_f_keyword_indexed.txt", "../qnaData_all/comment_f_indexed.txt", 0.20,
                                        maxlen, max_caption_len)

codes, partial_captions, next_words, predict_words= data_gen.MakeDataset4(train=True, mem_size=mem_size)
#
codes = sequence.pad_sequences(codes, maxlen=maxlen)
partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_caption_len)
predict_words = sequence.pad_sequences(predict_words, maxlen=mem_size)

codesT, partial_captionsT, next_wordsT, predict_wordsT = data_gen.MakeDataset4(train=False, mem_size=mem_size)
codesT = sequence.pad_sequences(codesT, maxlen=maxlen)
partial_captionsT = sequence.pad_sequences(partial_captionsT, maxlen=max_caption_len)
predict_wordsT = sequence.pad_sequences(predict_wordsT, maxlen=mem_size)

vocab_size = 5000

max_features = 10000  # size of token

nb_output = data_gen.vocab_size  # len(Y_train[0])

nb_filter = 128
filter_length = 5
hidden_dims = 256
embedding_dims = 128

mem_size = 30



embedding_for_nse = Embedding(vocab_size,128,dropout=0.3)
language_input = Input(shape=(max_caption_len,))
language_embedding = embedding_for_nse(language_input)
vocab_input = Input(shape=(mem_size,))
vocab_embedding = embedding_for_nse(vocab_input)

match = keras.layers.merge(mode='dot',inputs=[language_embedding,vocab_embedding],dot_axes=[2,2])
match = Activation('softmax')(match)

mem_encoder = keras.layers.merge(mode='dot',inputs=[match,vocab_embedding],dot_axes=[2,1])
mem_encoder = Activation('relu')(mem_encoder) # mem : memory embedding






# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's encode this vector sequence into a single vector
h = LSTM(400, return_sequences=True)(language_embedding)
# which will be used to compute a probability
# distribution over what the next word in the caption should be!


code_input = Input(shape=(maxlen,))
embedding_for_code = Embedding(max_features, 400, dropout=0.5)(code_input)

code_match = keras.layers.merge(mode='dot',inputs=[h,embedding_for_code],dot_axes=[2,2])
code_match = Activation('softmax')(code_match)
code_attention = keras.layers.merge(mode='dot',inputs=[code_match,embedding_for_code],dot_axes=[2,1])
code_attention = Activation('relu')(code_attention) # mem : memory embedding



code_model = Convolution1D(nb_filter=nb_filter,
                             filter_length=10,
                             border_mode='valid',
                             subsample_length=1)(code_attention)
code_model = Activation('relu')(code_attention)

code_model = Convolution1D(nb_filter=nb_filter,
                             filter_length=5,
                             border_mode='valid',
                             subsample_length=1)(code_model)
code_model = Activation('relu')(code_model)

code_model = Convolution1D(nb_filter=nb_filter,
                             filter_length=3,
                             border_mode='valid',
                             subsample_length=1)(code_model)
code_model = Activation('relu')(code_model)

code_model = MaxPooling1D(pool_length=code_model._keras_shape[1])(code_model)
code_model = Flatten()(code_model)
# We add a vanilla hidden layer:
code_model = Dense(400)(code_model)


t = Dense(400)(code_attention)

h = Dense(400)(h)






x = keras.layers.merge(mode='concat',inputs=[t,h,mem_encoder], concat_axis=-1)
x = LSTM(400, return_sequences=False)(x)

x = Dense(400)(x)

x = Dropout(0.5)(x)
predictions = Dense(nb_output,activation='softmax')(x)
optimizer = RMSprop(lr=0.0001)
model = Model(input=[code_input,language_input,vocab_input], output=predictions)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])


model.summary()
print "train"

codesL, commentsL, raw_commentL = data_gen.getTestData()

del data_gen

epoch = 200

for i in xrange(epoch):
    model.fit([codes, partial_captions,predict_words], next_words, batch_size=128, nb_epoch=1,
              validation_data=([codesT, partial_captionsT,predict_wordsT], next_wordsT))

json_string = model.to_json()
open('../model/attmem_qnaData_all.json', 'w').write(json_string)
model.save_weights('../model/attmem_qnaData_all.h5')








