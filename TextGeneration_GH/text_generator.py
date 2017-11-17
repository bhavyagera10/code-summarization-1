from keras.models import model_from_json, Sequential
from keras.preprocessing import sequence
import numpy as np
import math
import logging
import sys
import time
import nltk
import data_generator
from scipy.stats.mstats_basic import sen_seasonal_slopes
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation, Flatten, Dropout, RepeatVector
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import RMSprop
import data_generator
from keras.preprocessing import sequence
import keras
from keras.layers import Input, LSTM, Embedding, Dense, ZeroPadding1D
from keras.models import Model,Sequential
from keras.layers.wrappers import TimeDistributed
from custom_keras_layers.attention_layers_level1 import level1_attention


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
def make_model():
    max_caption_len = 20
    maxlen = 600
    mem_size = 30
    vocab_size = 4000

    max_features = 6000  # size of token

    #nb_output = data_gen.vocab_size  # len(Y_train[0])
    nb_output = 3917  # len(Y_train[0])

    nb_filter = 128
    filter_length = 10
    hidden_dims = 256
    embedding_dims = 128

    mem_size = 30


    language_input = Input(shape=(max_caption_len,))
    embedding_for_language = Embedding(vocab_size, 128, dropout=0.3)(language_input)
    language_embedding = LSTM(output_dim=128, return_sequences=True)(embedding_for_language)
    language_embedding = TimeDistributed(Dense(128))(language_embedding)

    code_input = Input(shape=(maxlen,))
    embedding_for_cnn = Embedding(max_features, 128, dropout=0.3)(code_input)
    attention_layer = level1_attention(layers=[embedding_for_cnn, embedding_for_language], name='Attention Layer')

    zeropadding_quest = ZeroPadding1D(filter_length - 1)(attention_layer)

    code_model = Convolution1D(nb_filter=nb_filter,
                               filter_length=10,
                               border_mode='valid',
                               subsample_length=1)(zeropadding_quest)
    code_model = Activation('relu')(code_model)

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
    code_model = Dense(256)(code_model)

    code_model = RepeatVector(max_caption_len)(code_model)

    # the output of both models will be tensors of shape (samples, max_caption_len, 128).
    # let's concatenate these 2 vector sequences.
    x = keras.layers.merge(mode='concat', inputs=[code_model, language_embedding], concat_axis=-1)
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

class SentenceGeneration(object):
    def __init__(self):
        self.model = Sequential()
        self.index2word = dict()
        self.word2Index = dict()
        self.index2token = dict()
        self.token2Index = dict()
        with open("../data/comment_f_Vocab.txt") as fin:
            for i, word in enumerate(fin):
                word = word.rstrip()
                self.index2word[i] = word
                self.word2Index[word] = int(i)
        with open("../data/code_f_Vocab.txt") as fin:
            for i, word in enumerate(fin):
                word = word.rstrip()
                self.index2token[i] = word
                self.token2Index[word] = int(i)

    # def __init__(self,codes,targets):
    #     self.model = Sequential()
    #     # self.codes = codes
    #     # self.targets = targets

    def readModel(self, name):
        #model = model_from_json(open('../model/'+name + '.json').read())
        model = make_model()
        model.load_weights('../model/'+name + '.h5')
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

        self.model = model

    def printSentence(self, indices):
        return ' '.join([self.index2word[x] for x in indices])

        #for index in indices:
        #    print self.index2word[index],
    def returnCode(self, indices):

        return ' '.join([self.index2token[x-1] for x in indices])

    def removeToken(self,indices):
        indices = indices[1:-1]
        unk_index = self.word2Index['UNK']
        if unk_index in indices:
            indices.remove(unk_index)
        return indices





    def generateSentence(self, code,beam_size):

        #test_code = [[47,8,528,529,128,78,79,12,81,7,8,9,81,17,18,78,22,12,81,7,81,17,30,72,83,30,36]]
        code = sequence.pad_sequences([code], maxlen=600)

        # target START,the,index,name,to,use,when,UNK,the,shard,to,explain,END
        # test_code = np.array(test_code)
        # test_code = np.reshape(test_code, (1, 400), (1))
        start_index = self.word2Index['START']
        end_index = self.word2Index['END']
        caption = [[start_index]]
        caption = sequence.pad_sequences(caption, maxlen=20)

        pred = self.model.predict([code, caption])[0]
        sorted_pred = np.argsort(pred)[::-1]

        # initial
        beams = []
        for index in sorted_pred[:beam_size]:
            beams.append((math.log(pred[index]), [start_index, index]))

        finished = []

        for t in range(1, 20):  # max_caption_len
            # generate
            candidates = []
            for b in beams:
                caption = sequence.pad_sequences([b[1]], maxlen=20)
                pred = self.model.predict([code, caption])[0]
                sorted_pred = np.argsort(pred)[::-1]

                for index in sorted_pred[:beam_size]:
                    new_caption = b[1][:]
                    new_caption.append(index)
                    candidates.append([b[0] + math.log(pred[index]), new_caption])
            candidates.sort(reverse=True)
            beams = candidates[:beam_size]

            # end sentence
            unfinished = []
            for b in beams:
                if b[1][-1] == end_index:
                    finished.append(b)
                    beam_size -= 1
                else:
                    unfinished.append(b)
            beams = unfinished[:]

            # print ""
            # for b in beams:
            #     print "%5f" % b[0], ' '.join([self.index2word[key] for key in b[1]])

            if beam_size == 0:
                break

        for i in range(beam_size):
            finished.append(beams[i])

        # re-ranking
        for f in finished:
            f[0] = f[0] / len(f[1])
            finished.sort(reverse=True)

        # print "---generate sentences---"
        s = []
        for b in finished:
            #sen = ' '.join([self.index2word[key] for key in b[1]])
            s.append(b[1])
        return s



gen = SentenceGeneration()
gen.readModel('attCNN')

data_gen = data_generator.DataGenerator("../data/code_f_indexed.txt", "../data/comment_f_indexed.txt",
                                        0.20, 600, 20)

codes,comments,raw_comment= data_gen.getTestData()

np.random.seed(30)
np.random.shuffle(codes)
np.random.seed(30)
np.random.shuffle(raw_comment)
np.random.seed(30)
np.random.shuffle(comments)

sum_of_BLEU1 = 0
sum_of_BLEU2 = 0
sum_of_BLEU3 = 0
sum_of_BLEU4 = 0
j = 0

sens = []
co = []
comm = []
raw_com=[]

score = []

for code,comment,raw in zip(codes,comments,raw_comment)[:]:
    sentences = gen.generateSentence(code,5)



    sens.append(sentences)
    co.append(gen.returnCode(code))
    comm.append(gen.printSentence(comment))
    raw_com.append(raw)
    #print "origen"
    #print gen.printSentence(comment)
    #print "generate"
    #for s in sentences:
    #    print gen.printSentence(s)
    #print comment
    #print sentences[0]
    # s = gen.removeToken(sentences[0])
    # c = gen.removeToken(comment)
    s = sentences[0][1:-1]
    c = comment[1:-1]

    BLEU1_score = nltk.translate.bleu_score.modified_precision([c], s, n=1)
    BLEU2_score = nltk.translate.bleu_score.modified_precision([c], s, n=2)
    BLEU3_score = nltk.translate.bleu_score.modified_precision([c], s, n=3)
    BLEU4_score = nltk.translate.bleu_score.modified_precision([c], s, n=4)

    sum_of_BLEU1 += float(BLEU1_score)
    sum_of_BLEU2 += float(BLEU2_score)
    sum_of_BLEU3 += float(BLEU3_score)
    sum_of_BLEU4 += float(BLEU4_score)
    score.append([BLEU1_score,j])
    j += 1


    sys.stdout.write('\r' + str(j)+' score :'+str(sum_of_BLEU1 / float(j)))


    #print 'score :', sum_of_BLEU / float(i)
    #print i,float(BLEU_score)




score.sort(reverse=True)
#high_score_index = [index[1] for index in score]

with open('attCNN_rand.txt', 'w') as fin:
    fin.write(str(sum_of_BLEU1 / float(j)))
    fin.write("\n")
    fin.write(str(sum_of_BLEU2 / float(j)))
    fin.write("\n")
    fin.write(str(sum_of_BLEU3 / float(j)))
    fin.write("\n")
    fin.write(str(sum_of_BLEU4 / float(j)))
    fin.write("\n")
    for i in range(len(co[:])):
        s = str(co[i])
        code = s.replace("{","{\n").replace(";",";\n").replace("}","}\n")
        fin.write("code:\n"+ code)
        fin.write("comment:\n"+ comm[i].rstrip()+ '\n')
        fin.write("--generate--\n")
        for s in sens[i]:
            fin.write(gen.printSentence(s[1:-1])+'\n')
        fin.write("\n")

with open('attCNN_sort.txt', 'w') as fin:
    fin.write(str(sum_of_BLEU1 / float(j)))
    fin.write("\n")
    fin.write(str(sum_of_BLEU2 / float(j)))
    fin.write("\n")
    fin.write(str(sum_of_BLEU3 / float(j)))
    fin.write("\n")
    fin.write(str(sum_of_BLEU4 / float(j)))
    fin.write("\n")

    for (sc,i) in score:
        s = str(co[i])
        code = s.replace("{", "{\n").replace(";", ";\n").replace("}", "}\n")
        fin.write("code:\n" + code)
        fin.write("comment:\n" + comm[i].rstrip() + '\n')
        fin.write("BLUE1:"+ str(float(sc))+ '\n')
        fin.write("--generate--\n")
        for s in sens[i]:
            fin.write(gen.printSentence(s[1:-1]) + '\n')
        fin.write("\n")
