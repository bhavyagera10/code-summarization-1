import logging
import math
import sys

import nltk
import numpy as np
from keras.models import model_from_json, Sequential
from keras.preprocessing import sequence
import data_generator


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




logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
def make_model():
    max_caption_len = 20
    maxlen = 600
    mem_size = 30

    vocab_size = 4000

    max_features = 6000  # size of token

    nb_filter = 128
    filter_length = 5
    hidden_dims = 256
    embedding_dims = 128

    mem_size = 30

    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 128, input_length=max_caption_len, dropout=0.3))
    vocab_encoder = Sequential()
    vocab_encoder.add(Embedding(vocab_size, 128, input_length=mem_size, dropout=0.3))

    match = Sequential()
    match.add(Merge([language_model, vocab_encoder], mode='dot', dot_axes=[2, 2]))  # inner product
    match.add(Activation('softmax'))  # z : matching similarity
    # output: (samples, maxlen_of_summary, vocab_size)

    mem_encoder = Sequential()
    mem_encoder.add(Merge([match, vocab_encoder], mode='dot', dot_axes=[2, 1]))
    mem_encoder.add(Activation('relu'))  # mem : memory embedding

    # the output of both models will be tensors of shape (samples, max_caption_len, 128).
    # let's concatenate these 2 vector sequences.
    model = Sequential()
    model.add(Merge([language_model, mem_encoder], mode='concat', concat_axis=-1))
    # let's encode this vector sequence into a single vector
    model.add(LSTM(256, return_sequences=False))
    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    model.add(Dropout(0.5))
    model.add(Dense(3917))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.0001)

    return model

class SentenceGeneration(object):
    def __init__(self):
        self.model = Sequential()
        self.index2word = dict()
        self.word2Index = dict()
        self.index2token = dict()
        self.token2Index = dict()
        self.predict_model = data_generator.SentenceGeneration()
        self.predict_model.readModel('keyword_f')

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
        model = model_from_json(open('../model/'+name + '.json').read())
        #model = make_model()
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
        #code = sequence.pad_sequences([code], maxlen=600)
        #keyword_model = data_generator.SentenceGeneration()
        #keyword_model.readModel('keyword_f')

        c = sequence.pad_sequences([code], maxlen=600)
        skip = sequence.pad_sequences([[str(0)]], maxlen=600)
        keyword_pred = self.predict_model.model.predict(c)[0]

        keyword_sorted_pred = np.argsort(keyword_pred)[::-1]
        keywords = [s for s in keyword_sorted_pred[:30]]
        keywords = sequence.pad_sequences([keywords], maxlen=30)

        # target START,the,index,name,to,use,when,UNK,the,shard,to,explain,END
        # test_code = np.array(test_code)
        # test_code = np.reshape(test_code, (1, 400), (1))
        start_index = self.word2Index['START']
        end_index = self.word2Index['END']
        caption = [[start_index]]
        caption = sequence.pad_sequences(caption, maxlen=20)

        pred = self.model.predict([c,caption,keywords])[0]
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
                pred = self.model.predict([skip,caption,keywords])[0]
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
gen.readModel('fcode_words_server2')

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

with open('fcode_words_rand2.txt', 'w') as fin:
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

with open('fcode_words_sort2.txt', 'w') as fin:
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
