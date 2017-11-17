from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation, Flatten, Dropout, RepeatVector
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import RMSprop
import data_generator
from keras.preprocessing import sequence
import keras
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model,Sequential
from keras.layers.wrappers import TimeDistributed

max_caption_len = 26
maxlen = 500
mem_size = 30

data_gen = data_generator.DataGenerator("../qnaData/code_f_indexed.txt", "../qnaData/comment_f_indexed.txt", 0.20,
                                        maxlen, max_caption_len)

codes, partial_captions, next_words, predict_words= data_gen.MakeDataset5(train=True, mem_size=mem_size)
#
codes = sequence.pad_sequences(codes, maxlen=maxlen)
partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_caption_len)
predict_words = sequence.pad_sequences(predict_words, maxlen=mem_size)

codesT, partial_captionsT, next_wordsT, predict_wordsT = data_gen.MakeDataset5(train=False, mem_size=mem_size)
codesT = sequence.pad_sequences(codesT, maxlen=maxlen)
partial_captionsT = sequence.pad_sequences(partial_captionsT, maxlen=max_caption_len)
predict_wordsT = sequence.pad_sequences(predict_wordsT, maxlen=mem_size)

vocab_size = 4000

max_features = 6000  # size of token

nb_output = data_gen.vocab_size  # len(Y_train[0])

nb_filter = 128
filter_length = 5
hidden_dims = 256
embedding_dims = 128

mem_size = 30
class Models(object):


    def code_model(self):
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

        code_model.add(RepeatVector(max_caption_len))

        # next, let's define a RNN model that encodes sequences of words
        # into sequences of 128-dimensional word vectors.

        language_input = Input(shape=(max_caption_len,))
        language_embedding = Embedding(vocab_size, 128, dropout=0.3)(language_input)
        language_embedding = LSTM(output_dim=128, return_sequences=True)(language_embedding)
        language_embedding = TimeDistributed(Dense(128))(language_embedding)

        code_input = Input(shape=(maxlen,))
        code_embedding = code_model(code_input)

        # the output of both models will be tensors of shape (samples, max_caption_len, 128).
        # let's concatenate these 2 vector sequences.
        x = keras.layers.merge(mode='concat', inputs=[code_embedding, language_embedding], concat_axis=-1)
        # let's encode this vector sequence into a single vector
        x = LSTM(256, return_sequences=False)(x)
        # which will be used to compute a probability
        # distribution over what the next word in the caption should be!
        x = Dropout(0.5)(x)
        predictions = Dense(nb_output, activation='softmax')(x)

        optimizer = RMSprop(lr=0.0001)
        model = Model(input=[code_input, language_input], output=predictions)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        return model
    def words_model(self):
        embedding_for_nse = Embedding(vocab_size, 128, dropout=0.3)

        language_input = Input(shape=(max_caption_len,))
        language_embedding = embedding_for_nse(language_input)

        vocab_input = Input(shape=(mem_size,))
        vocab_embedding = embedding_for_nse(vocab_input)

        match = keras.layers.merge(mode='dot', inputs=[language_embedding, vocab_embedding], dot_axes=[2, 2])
        match = Activation('softmax')(match)

        # match.add(Merge([language_model,vocab_encoder],mode='dot', dot_axes=[2, 2])) #inner product'
        # match.add(Merge([language_embedding,vocab_embedding],mode='dot', dot_axes=[2, 2])) #inner product
        # match.add(Activation('softmax')) # z : matching similarity
        # output: (samples, maxlen_of_summary, vocab_size)

        # mem_encoder = Sequential()
        mem_encoder = keras.layers.merge(mode='dot', inputs=[match, vocab_embedding], dot_axes=[2, 1])
        mem_encoder = Activation('relu')(mem_encoder)  # mem : memory embedding

        # the output of both models will be tensors of shape (samples, max_caption_len, 128).
        # let's concatenate these 2 vector sequences.
        x = keras.layers.merge(mode='concat', inputs=[language_embedding, mem_encoder], concat_axis=-1)
        # let's encode this vector sequence into a single vector
        x = LSTM(256, return_sequences=False)(x)
        # which will be used to compute a probability
        # distribution over what the next word in the caption should be!
        x = Dropout(0.5)(x)
        predictions = Dense(nb_output, activation='softmax')(x)

        optimizer = RMSprop(lr=0.0001)

        model = Model(input=[language_input, vocab_input], output=predictions)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        return model

    def code_words_model(self):

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

        #code_model.add(RepeatVector(max_caption_len))

        # next, let's define a RNN model that encodes sequences of words
        # into sequences of 128-dimensional word vectors.

        embedding_for_nse = Embedding(vocab_size,128,dropout=0.3)

        language_input = Input(shape=(max_caption_len,))
        language_embedding = embedding_for_nse(language_input)

        vocab_input = Input(shape=(mem_size,))
        vocab_embedding = embedding_for_nse(vocab_input)

        match = keras.layers.merge(mode='dot',inputs=[language_embedding,vocab_embedding],dot_axes=[2,2])
        match = Activation('softmax')(match)

        mem_encoder = keras.layers.merge(mode='dot',inputs=[match,vocab_embedding],dot_axes=[2,1])
        mem_encoder = Activation('relu')(mem_encoder) # mem : memory embedding

        mlp_model = keras.layers.merge(mode='concat',inputs=[language_embedding,mem_encoder], concat_axis=-1)
        mlp_model = Dense(128)(mlp_model)
        mlp_model = Dropout(0.3)(mlp_model)



        code_input = Input(shape=(maxlen,))
        code_embedding = code_model(code_input)
        code_embedding = RepeatVector(max_caption_len)(code_embedding)



        # the output of both models will be tensors of shape (samples, max_caption_len, 128).
        # let's concatenate these 2 vector sequences.
        x = keras.layers.merge(mode='concat',inputs=[code_embedding,mlp_model], concat_axis=-1)
        # let's encode this vector sequence into a single vector
        x = LSTM(256, return_sequences=False)(x)
        # which will be used to compute a probability
        # distribution over what the next word in the caption should be!
        x = Dropout(0.5)(x)
        predictions = Dense(nb_output,activation='softmax')(x)

        optimizer = RMSprop(lr=0.0001)
        model = Model(input=[code_input,language_input, vocab_input], output=predictions)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        return model

models = Models()
model = models.code_words_model()
model.summary()

print "train"
early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='auto')

del data_gen

#model.fit([codes,partial_captions,predict_words], next_words, batch_size=128, nb_epoch=200,
#          validation_data=([codesT,partial_captionsT,predict_wordsT], next_wordsT), callbacks=[early])
model.fit([codes,partial_captions,predict_words], next_words, batch_size=128, nb_epoch=200,
          validation_data=([codesT,partial_captionsT,predict_wordsT], next_wordsT), callbacks=[early])

json_string = model.to_json()
open('../model/fcode_words_server2.json', 'w').write(json_string)
model.save_weights('../model/fcode_words_server2.h5')
