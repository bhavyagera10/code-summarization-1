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
import data_generator
from keras.preprocessing import sequence
import keras


max_caption_len = 26
maxlen = 500

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
#code_model.add(Dropout(0.5))
code_model.add(Dense(256))
# next, let's define a RNN model that encodes sequences of words
# into sequences of 128-dimensional word vectors.
language_model = Sequential()
language_model.add(Embedding(vocab_size, 128, input_length=max_caption_len, dropout=0.2))
'''
language_model.add(LSTM(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))
'''
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
model.add(Dropout(0.5))
model.add(Dense(nb_output))
model.add(Activation('softmax'))


optimizer = RMSprop(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

print "train"
early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
cp = keras.callbacks.ModelCheckpoint('../model/gen_f2.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')


del data_gen
 
model.fit([codes, partial_captions], next_words, batch_size=128, nb_epoch=200, validation_data=([codesT, partial_captionsT], next_wordsT), callbacks=[early])

json_string = model.to_json()
open('../model/q_code3.json', 'w').write(json_string)
model.save_weights('../model/q_code3.h5')
