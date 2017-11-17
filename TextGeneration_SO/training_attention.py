
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True))

from keras.models import Sequential
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation, Flatten, Dense, Dropout, RepeatVector
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import RMSprop
from keras.layers import Input, LSTM, Embedding, Dense, ZeroPadding1D
from custom_keras_layers.attention_layers_level1 import level1_attention
from custom_keras_layers.attention_layers_level2 import level2_attention
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




# next, let's define a RNN model that encodes sequences of words
# into sequences of 128-dimensional word vectors.

embedding_for_nse = Embedding(vocab_size,128,dropout=0.3)
language_input = Input(shape=(max_caption_len,))
language_embedding = embedding_for_nse(language_input)
memory_input = Input(shape=(mem_size,))
memory_embedding = embedding_for_nse(memory_input)

match = keras.layers.merge(mode='dot',inputs=[language_embedding,memory_embedding],dot_axes=[2,2])
match = Activation('softmax')(match)


mem_encoder = keras.layers.merge(mode='dot',inputs=[match,memory_embedding],dot_axes=[2,1])
mem_encoder = Activation('relu')(mem_encoder) # mem : memory embedding





code_input = Input(shape=(maxlen,))
embedding_for_cnn = Embedding(max_features, 128, dropout=0.3)(code_input)
attention_layer = level1_attention(layers=[embedding_for_cnn, language_embedding], name='Attention Layer')

attention2_code = level2_attention(1, layers=[embedding_for_cnn, attention_layer], axes=2)
zeropadding_code = ZeroPadding1D(filter_length - 1)(attention2_code)
conv_code = Convolution1D(nb_filter=nb_filter,
                             filter_length=10,
                             border_mode='valid',activation='relu',
                             subsample_length=1)(zeropadding_code)


conv_code = Convolution1D(nb_filter=nb_filter,
                             filter_length=5,
                             border_mode='valid',activation='relu',
                             subsample_length=1)(conv_code)

conv_code = Convolution1D(nb_filter=nb_filter,
                             filter_length=3,
                             border_mode='valid',activation='relu',
                             subsample_length=1)(conv_code)

maxpool_code = MaxPooling1D(pool_length=(conv_code._keras_shape[1]),)(conv_code)
flat_code = Flatten()(maxpool_code)


code_model = Dense(256)(flat_code)

code_model = RepeatVector(max_caption_len)(code_model)





# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
x = keras.layers.merge(mode='concat',inputs=[code_model,language_embedding, mem_encoder], concat_axis=-1)
# let's encode this vector sequence into a single vector
x = LSTM(256, return_sequences=False)(x)
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
x = Dropout(0.5)(x)
predictions = Dense(nb_output,activation='softmax')(x)

optimizer = RMSprop(lr=0.0001)
model = Model(input=[code_input,language_input, memory_input], output=predictions)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

print "train"

codesL, commentsL, raw_commentL = data_gen.getTestData()
ids = data_gen.getTestID()

del data_gen


max_score = 0

epoch = 100

for i in xrange(epoch):
    model.fit([codes, partial_captions, predict_words], next_words, batch_size=128, nb_epoch=1,
              validation_data=([codesT, partial_captionsT, predict_wordsT], next_wordsT))

    gen = SentenceGeneration()
    gen.setModel(model)
    sum_of_BLEU1 = 0

    sens = []
    co = []
    comm = []
    raw_com = []

    score = []
    j = 0
    for code, comment, raw in zip(codesL, commentsL, raw_commentL)[:]:
        sentences = gen.generateSentence(code, 10,nonUNK=False,useMemory=True)


        sens.append(sentences)
        co.append(gen.returnCode(code))


    ids = []
    with open('../qnaData/codenn.txt') as f:
        ids = [val.split('\t')[0] for val in f]

    reference_file = "ref_UNK.txt"
    predictions = []
    for i, id in enumerate(ids):
        predictions.append(str(id) + "\t" + gen.printSentence(sens[i][0][1:-1]) + '\n')

    (goldMap, predictionMap) = computeMaps(predictions, reference_file)
    score = bleuFromMaps(goldMap, predictionMap)[0]


    print 'eter:'+str(i)
    print score

    if max_score < score:
        max_score = score
        json_string = model.to_json()
        open('../model/q_mem_att.json', 'w').write(json_string)
        model.save_weights('../model/q_mem_att.h5')






