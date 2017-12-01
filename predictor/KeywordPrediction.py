import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True))


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



logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class SentenceGeneration(object):
    def __init__(self):
        self.model = Sequential()
        self.index2word = dict()
        self.word2Index = dict()
        self.index2token = dict()
        self.token2Index = dict()
        with open("../data/comment_f_keyword_Vocab.txt") as fin:
            for i, word in enumerate(fin):
                word = word.rstrip()
                self.index2word[i] = word
                self.word2Index[word] = int(i)
        with open("../data/code_f_keyword_Vocab.txt") as fin:
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





    def generateSentence(self, code, n):

        code = sequence.pad_sequences([code], maxlen=600)


        pred = self.model.predict(code)[0]
        sorted_pred = np.argsort(pred)[::-1]

        pred_word = [ self.index2token[index] for index in sorted_pred[:n]]
        return pred_word


def rAtk(pred, target, k):
    #sorted_pred = np.argsort(pred)[::-1]
    correct = 0
    for i in pred[:k]:
        if i in target:
            correct += 1

    return correct / float(len(target))

gen = SentenceGeneration()
gen.readModel('keyword_f')

data_gen = data_generator.DataGenerator("../data/code_f_keyword_indexed.txt", "../data/comment_f_keyword_indexed.txt",
                                        0.20, 600, 20)

codes,keywords,raw_comment= data_gen.getTestData()

np.random.seed(30)
np.random.shuffle(codes)
np.random.seed(30)
np.random.shuffle(raw_comment)
np.random.seed(30)
np.random.shuffle(keywords)


sens = []
co = []
comm = []
r = 0
k = 30
for i,(code,comment) in enumerate(zip(codes,keywords)):
    #keyword = gen.generateSentence(code,5)
    c = sequence.pad_sequences([code], maxlen=600)
    pred = gen.model.predict(c)[0]
    #pred[52] = 0

    sorted_pred = np.argsort(pred)[::-1]

    sens.append([gen.index2word[s] for s in sorted_pred[:k]])
    co.append(gen.returnCode(code))
    comm.append(gen.printSentence(comment))

    r += rAtk(sorted_pred, comment, k)



    sys.stdout.write('\r' + str(i)+' score :'+str(r/(i+1)))






with open('keyword_prediction4.txt', 'w') as fin:
    fin.write(str(r/(i+1)))
    fin.write("\n")
    for i in range(len(co[:300])):
        s = str(co[i])
        code = s.replace("{","{\n").replace(";",";\n").replace("}","}\n")
        fin.write("code:\n"+ code)
        fin.write("comment:\n"+ raw_comment[i].rstrip()+ '\n')
        fin.write("--generate--\n")

        fin.write(' '.join(sens[i])+'\n')
        fin.write("\n")
    # for sen in sens:
    #     fin.write(gen.printSentence(sen[1:-1])+'\n')
