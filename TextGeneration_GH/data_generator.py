import numpy as np

class DataGenerator(object):
    
    def __init__(self,code_path,commnet_path,test_persentage,max_code_len, max_commnet_len):
        self.raw_code = []
        self.raw_comment = []

        self.code_data = []
        self.comment_data = []
        
        self.train_code_data = []
        self.train_comment_data = []
        
        self.test_code_data = []
        self.test_comment_data = []
        self.test_raw_comment = []
        
        self.index2word = dict()
        
        self.ReadData(code_path, commnet_path,max_code_len, max_commnet_len)
        self.SplitData(test_persentage)
        
        
        self.vocab_size = len(self.index2word)
    def getTestData(self):
        return self.test_code_data,self.test_comment_data, self.test_raw_comment

    def ReadData(self, code_path, commnet_path, max_code_len, max_commnet_len):
        code_lines=[]
        comment_lines=[]
        raw_comment_lines = []
        with open(code_path) as fin:
            code_lines = fin.readlines()
        with open(commnet_path) as fin:
            comment_lines = fin.readlines()
        with open('../data/comment_preproc_F.txt') as fin:
            raw_comment_lines = fin.readlines()

        for code_line, comment_line, raw_comment_line in zip(code_lines,comment_lines,raw_comment_lines):
            tokens = code_line.split(",")
            words = comment_line.split(",")
            if len(tokens) > max_code_len or len(words) > max_commnet_len:
                continue
            if len(tokens) <= 12 or len(words) <= 3:
                continue
            tokens = [int(token) for token in tokens]
            self.code_data.append(tokens)
            words = [int(token) for token in words]
            self.comment_data.append(words)

            self.raw_comment.append(raw_comment_line)



        
        np.random.seed(30)
        np.random.shuffle(self.code_data)
        
        np.random.seed(30)
        np.random.shuffle(self.comment_data)

        np.random.seed(30)
        np.random.shuffle(self.raw_comment)
        print "num of data:",len(self.code_data)
        
        with open("../data/comment_f_Vocab.txt") as fin:
            for i, word in enumerate(fin):
                self.index2word[i] = word.rstrip() 

                
    def SplitData(self,test_persentage):
        
        self.train_code_data = self.code_data[:int(len(self.code_data) * (1 - test_persentage))]
        self.train_comment_data = self.comment_data[:int(len(self.comment_data) * (1 - test_persentage))]
        
        self.test_code_data = self.code_data[int(len(self.code_data) * (1 - test_persentage)):]
        self.test_comment_data = self.comment_data[int(len(self.comment_data) * (1 - test_persentage)):]
        self.test_raw_comment = self.raw_comment[int(len(self.raw_comment) * (1 - test_persentage)):]
                
        
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