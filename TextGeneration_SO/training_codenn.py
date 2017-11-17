
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

data_gen = data_generator.DataGenerator("../qnaData/code_f_keyword_indexed.txt", "../qnaData/comment_f_indexed.txt", 0.20,
                                        maxlen, max_caption_len)

codes, partial_captions, next_words = data_gen.MakeDataset3(train=True)
#
codes = sequence.pad_sequences(codes, maxlen=maxlen)
partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_caption_len)

codesT, partial_captionsT, next_wordsT = data_gen.MakeDataset3(train=False)
codesT = sequence.pad_sequences(codesT, maxlen=maxlen)
partial_captionsT = sequence.pad_sequences(partial_captionsT, maxlen=max_caption_len)

vocab_size = 5000

max_features = 10000  # size of token

nb_output = data_gen.vocab_size  # len(Y_train[0])

nb_filter = 128
filter_length = 5
hidden_dims = 512
embedding_dims = 512

mem_size = 30




embedding_for_nse = Embedding(vocab_size,embedding_dims,dropout=0.3)
language_input = Input(shape=(max_caption_len,))
language_embedding = embedding_for_nse(language_input)


# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's encode this vector sequence into a single vector
h = LSTM(hidden_dims, return_sequences=True)(language_embedding)
# which will be used to compute a probability
# distribution over what the next word in the caption should be!


code_input = Input(shape=(maxlen,))
embedding_for_code = Embedding(max_features, hidden_dims, dropout=0.5)(code_input)

code_match = keras.layers.merge(mode='dot',inputs=[h,embedding_for_code],dot_axes=[2,2])
code_match = Activation('softmax')(code_match)
code_attention = keras.layers.merge(mode='dot',inputs=[code_match,embedding_for_code],dot_axes=[2,1])
code_attention = Activation('relu')(code_attention) # mem : memory embedding




t = Dense(hidden_dims)(code_attention)

h = Dense(hidden_dims)(h)

x = keras.layers.merge(mode='concat',inputs=[t,h], concat_axis=-1)
x = LSTM(hidden_dims, return_sequences=False)(x)

x = Dense(hidden_dims)(x)

x = Dropout(0.5)(x)
predictions = Dense(nb_output,activation='softmax')(x)
optimizer = RMSprop(lr=0.0001)
model = Model(input=[code_input,language_input], output=predictions)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])


model.summary()
print "train"

codesL, commentsL, raw_commentL = data_gen.getTestData()
ids = data_gen.getTestID()

del data_gen


max_score = 0

epoch = 200

ids = []
with open('../qnaData/codenn.txt') as f:
    ids = [val.split('\t')[0] for val in f]


for i in xrange(epoch):
    model.fit([codes, partial_captions], next_words, batch_size=128, nb_epoch=1,
              validation_data=([codesT, partial_captionsT], next_wordsT))

    gen = SentenceGeneration()
    gen.setModel(model)
    sum_of_BLEU1 = 0

    sens = []
    co = []


    score = []

    for code, comment, raw in zip(codesL, commentsL, raw_commentL)[:]:
        sentences = gen.generateSentence(code, 5,nonUNK=0,useMemory=False)


        sens.append(sentences)
        co.append(gen.returnCode(code))




    reference_file = "ref_UNK.txt"
    predictions = []
    for j, id in enumerate(ids):
        predictions.append(str(id) + "\t" + gen.printSentence(sens[j][0][1:-1]) + '\n')

    del gen

    (goldMap, predictionMap) = computeMaps(predictions, reference_file)
    score = bleuFromMaps(goldMap, predictionMap)[0]

    print 'eter'+str(i)
    print score

    if max_score < score and i>5:
        max_score = score
        json_string = model.to_json()
        open('../model/codenn_512.json', 'w').write(json_string)
        model.save_weights('../model/codenn_512.h5')

    print max_score






