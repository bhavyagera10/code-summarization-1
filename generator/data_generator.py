import numpy as np
from keras.models import model_from_json, Sequential
from keras.preprocessing import sequence

class SentenceGeneration(object):
    def __init__(self):
        self.model = Sequential()
        self.index2word = dict()
        self.word2Index = dict()
        self.index2token = dict()
        self.token2Index = dict()
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




class DataGenerator(object):
    def __init__(self, code_path, commnet_path, test_persentage, max_code_len, max_commnet_len):
        self.raw_code = []
        self.raw_comment = []

        self.code_split_plag = []

        self.code_data = []
        self.comment_data = []

        self.train_code_data = []
        self.train_comment_data = []

        self.valid_code_data = []
        self.valid_comment_data = []

        self.test_code_data = []
        self.test_comment_data = []
        self.test_raw_comment = []

        self.index2word = dict()

        self.ReadData(code_path, commnet_path, max_code_len, max_commnet_len)
        self.SplitData(test_persentage)

        self.vocab_size = len(self.index2word)

    def getTestData(self):
        return self.test_code_data, self.test_comment_data, self.test_raw_comment

    def getTestID(self):
        with open('../qnaData/codenn.txt') as f:
            ids = [val.split('\t')[0] for val in f]
        return ids

    def ReadData(self, code_path, commnet_path, max_code_len, max_commnet_len):
        code_lines = []
        comment_lines = []
        raw_comment_lines = []
        code_split_plags = []
        with open(code_path) as fin:
            code_lines = fin.readlines()
        with open(commnet_path) as fin:
            comment_lines = fin.readlines()
        with open('../qnaData/comment_preproc_F.txt') as fin:
            raw_comment_lines = fin.readlines()

        with open('../qnaData/qnaSet.txt') as fin:
            code_split_plags = fin.readlines()

        count = 0
        for code_line, comment_line, raw_comment_line, plag in zip(code_lines, comment_lines, raw_comment_lines,
                                                                   code_split_plags):
            tokens = code_line.split(",")
            words = comment_line.split(",")
            # if len(tokens) > max_code_len or len(words) > max_commnet_len:
            #    continue

            if (len(tokens) <= 12 or len(words) <= 3) and plag.rstrip() != 'test':
                continue

            tokens = [int(token) for token in tokens]
            self.code_data.append(tokens)
            words = [int(token) for token in words]
            self.comment_data.append(words)

            self.raw_comment.append(raw_comment_line)
            self.code_split_plag.append(str(plag).rstrip())

        # np.random.seed(30)
        # np.random.shuffle(self.code_data)

        # np.random.seed(30)
        # np.random.shuffle(self.comment_data)

        # np.random.seed(30)
        # np.random.shuffle(self.raw_comment)
        print "num of data:", len(self.code_data)

        with open("../qnaData/comment_f_Vocab.txt") as fin:
            for i, word in enumerate(fin):
                self.index2word[i] = word.rstrip()

    def SplitData(self, test_persentage):

        for code, comment, rawComment, flag in zip(self.code_data, self.comment_data, self.raw_comment,
                                                   self.code_split_plag):
            if flag == "train":
                self.train_code_data.append(code)
                self.train_comment_data.append(comment)
            elif flag == "valid":
                self.valid_code_data.append(code)
                self.valid_comment_data.append(comment)
            else:
                self.test_code_data.append(code)
                self.test_comment_data.append(comment)
                self.test_raw_comment.append(rawComment)


    def MakeDataset(self,train,divide=1,part=0):

        codes = []
        captions = []

        if train == True:
            c = len(self.train_code_data)/divide
            if divide == part+1:
                codes = self.train_code_data[c*part:]
                captions = self.train_comment_data[c*part:]
            else:
                codes = self.train_code_data[c * part:c * (part+1)]
                captions = self.train_comment_data[c * part:c * (part+1)]
        else:
            codes = self.test_code_data
            captions = self.test_comment_data

        train_codes = []
        patial_captions = []
        next_words = []


        for code, caption in zip(codes,captions):
            for i in range(1,len(caption)):
                next_word = [0 for j in range(self.vocab_size)]
                patial_caption = caption[:i]
                next_word[int(caption[i])] = 1

                train_codes.append(code)
                patial_captions.append(patial_caption)
                next_words.append(next_word)

        return train_codes, patial_captions, next_words

    def MakeDataset3(self, train, divide=1, part=0):

        codes = []
        captions = []

        if train:
            codes = self.train_code_data
            captions = self.train_comment_data

        else:
            codes = self.test_code_data
            captions = self.test_comment_data

        train_codes = []
        patial_captions = []
        next_words = []

        for code, caption in zip(codes, captions):
            for i in range(1, len(caption)):
                patial_caption = caption[:i]
                next_words.append(int(caption[i]))
                train_codes.append(code)
                patial_captions.append(patial_caption)



        next_words2 = np.zeros((len(next_words), self.vocab_size), dtype=np.bool)
        for i,index in enumerate(next_words):
            next_words2[i,index] = 1

        return train_codes, patial_captions, next_words2

    def MakeDataset4(self, train, mem_size):

        codes = []
        captions = []

        if train:
            codes = self.train_code_data
            captions = self.train_comment_data

        else:
            codes = self.test_code_data
            captions = self.test_comment_data

        train_codes = []
        patial_captions = []
        next_words = []

        gen = SentenceGeneration()
        gen.readModel('qnaData_all_keyword')
        predict_keywords = []

        for code, caption in zip(codes, captions):
            c = sequence.pad_sequences([code], maxlen=500)
            pred = gen.model.predict(c)[0]

            sorted_pred = np.argsort(pred)[::-1]

            for i in range(1, len(caption)):
                patial_caption = caption[:i]
                next_words.append(int(caption[i]))

                train_codes.append(code)
                patial_captions.append(patial_caption)
                predict_keywords.append([s for s in sorted_pred[:mem_size]])

        next_words2 = np.zeros((len(next_words), self.vocab_size), dtype=np.bool)
        for i,index in enumerate(next_words):
            next_words2[i, index] = 1

        return train_codes, patial_captions, next_words2, predict_keywords

    def MakeDataset2(self, train):
        codes = []
        captions = []

        if train:

            codes = self.train_code_data
            captions = self.train_comment_data
        else:
            codes = self.test_code_data
            captions = self.test_comment_data

        train_codes = []
        patial_captions = []


        for j,(code, caption) in enumerate(zip(codes, captions)):
            for i in range(1, len(caption)):


                patial_caption = caption[:i]

                train_codes.append(code)
                patial_captions.append(patial_caption)

        next_words = np.zeros((len(train_codes), self.vocab_size), dtype=np.bool)

        for j,(code, caption) in enumerate(zip(codes, captions)):
            for i in range(1, len(caption)):

                next_words[j, int(caption[i])] = 1

        return train_codes, patial_captions, next_words

     
                
                
             



'''
X_train = X_data[:int(len(X_data) * (1 - 0.25))]
Y_train = Y_data[:int(len(X_data) * (1 - 0.25))]

X_test = X_data[int(len(X_data) * (1 - 0.25)):]
Y_test = Y_data[int(len(X_data) * (1 - 0.25)):]

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
'''