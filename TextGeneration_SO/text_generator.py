import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True))

from keras.models import model_from_json, Sequential
import numpy as np
import math
import logging
import sys
import nltk
from keras.models import Sequential
import data_generator
from keras.preprocessing import sequence


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)




class SentenceGeneration(object):
    def __init__(self):
        self.model = Sequential()
        self.index2word = dict()
        self.word2Index = dict()
        self.index2token = dict()
        self.token2Index = dict()

        self.predict_model = self.readKeywordModel('q_keyword2')


        with open("../qnaData/comment_f_Vocab.txt") as fin:
            for i, word in enumerate(fin):
                word = word.rstrip()
                self.index2word[i] = word
                self.word2Index[word] = int(i)
        with open("../qnaData/code_f_keyword_Vocab.txt") as fin:
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

    def readKeywordModel(self, name):
        model = model_from_json(open('../model/' + name + '.json').read())
        model.load_weights('../model/' + name + '.h5')
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

        return model

    def setModel(self, model):
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





    def generateSentence(self, code,beam_size,nonUNK,useMemory):

        code = sequence.pad_sequences([code], maxlen=500)

        if useMemory:
            keyword_pred = self.predict_model.model.predict(code)[0]

            keyword_sorted_pred = np.argsort(keyword_pred)[::-1]
            keywords = [s for s in keyword_sorted_pred[:30]]
            keywords = sequence.pad_sequences([keywords], maxlen=30)


        start_index = self.word2Index['START']
        end_index = self.word2Index['END']
        caption = [[start_index]]
        caption = sequence.pad_sequences(caption, maxlen=26)

        if useMemory:
            pred = self.model.predict([code, caption, keywords])[0]
        else:
            pred = self.model.predict([code, caption])[0]
        sorted_pred = np.argsort(pred)[::-1]

        # initial
        beams = []
        for index in sorted_pred[:beam_size]:
            beams.append((math.log(pred[index]), [start_index, index]))

        finished = []

        for t in range(1, 26):  # max_caption_len
            # generate
            candidates = []
            for b in beams:
                caption = sequence.pad_sequences([b[1]], maxlen=26)
                if useMemory:
                    pred = self.model.predict([code, caption, keywords])[0]
                else:
                    pred = self.model.predict([code, caption])[0]
                sorted_pred = np.argsort(pred)[::-1]

                if nonUNK:
                    candi_index = [val for val in sorted_pred[:beam_size]]
                    if self.word2Index['UNK'] in candi_index:
                        candi_index.append(sorted_pred[beam_size+1])
                else:
                    candi_index = sorted_pred[:beam_size]


                for index in candi_index:
                    if nonUNK:
                        if self.word2Index['UNK'] == index:
                            continue

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



if __name__ == '__main__':
    gen = SentenceGeneration()
    gen.readModel('codenn_512')
    name = 'codenn_512'
    nonUNK = 1
    beamSize = 10
    useMemory = 1

    data_gen = data_generator.DataGenerator("../qnaData/code_f_keyword_indexed.txt", "../qnaData/comment_f_indexed.txt",
                                            0.20, 500, 26)

    codes,comments,raw_comment= data_gen.getTestData()
    '''
    np.random.seed(30)
    np.random.shuffle(codes)
    np.random.seed(30)
    np.random.shuffle(raw_comment)
    np.random.seed(30)
    np.random.shuffle(comments)
    '''
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
        sentences = gen.generateSentence(code,beamSize,nonUNK=nonUNK,useMemory=0)



        sens.append(sentences)
        co.append(gen.returnCode(code))
        comm.append(gen.printSentence(comment[1:-1]))
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

    with open('code_rand_'+name+'.txt', 'w') as fin:
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

    with open('code_sort_'+name+'.txt', 'w') as fin:
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


    with open('gold.txt', 'w') as fin:
        for i in range(len(co[:])):
            fin.write(str(i)+"\t"+comm[i].rstrip()+ '\n')
    with open('predict_n_'+name+'.txt', 'w') as fin:
        for i in range(len(co[:])):
            for s in sens[i]:
                fin.write(str(i)+"\t"+gen.printSentence(s[1:-1])+'\n')


    with open('predict_1_'+name+'.txt', 'w') as fin:
        for i in range(len(co[:])):
            fin.write(str(i)+"\t"+gen.printSentence(sens[i][0][1:-1])+'\n')
