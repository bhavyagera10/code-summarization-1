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
        self.gCodes = []
        self.gComment = []
        self.index2word = dict()
        
        self.ReadData(code_path, commnet_path,max_code_len, max_commnet_len)
        self.SplitData(test_persentage)
        
        
        self.vocab_size = len(self.index2word)
    def getTestData(self):
        return self.test_code_data,self.test_comment_data, self.test_raw_comment
    def getGdata(self):
        return self.gCodes,self.gComment
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

        gCodes = []
        gComment = []
        with open('../qnaData/code_f_keyword_indexed.txt') as f:
            for line in f:
                gCodes.append([str(c) for c in line.rstrip().split(',')])
        with open('../qnaData/comment_f_indexed.txt') as f:
            for line in f:
                gComment.append([str(c) for c in line.rstrip().split(',')])

        for code_line, comment_line, raw_comment_line,code,comment in zip(code_lines,comment_lines,raw_comment_lines,gCodes,gComment):
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
            self.gCodes.append(code)
            self.gComment.append(comment)



        
        np.random.seed(30)
        np.random.shuffle(self.code_data)
        
        np.random.seed(30)
        np.random.shuffle(self.comment_data)

        np.random.seed(30)
        np.random.shuffle(self.raw_comment)
        np.random.seed(30)
        np.random.shuffle(self.gCodes)
        np.random.seed(30)
        np.random.shuffle(self.gComment)
        print "num of data:",len(self.code_data)
        
        with open("../data/comment_f_keyword_Vocab.txt") as fin:
            for i, word in enumerate(fin):
                self.index2word[i] = word.rstrip() 

                
    def SplitData(self,test_persentage):
        
        self.train_code_data = self.code_data[:int(len(self.code_data) * (1 - test_persentage))]
        self.train_comment_data = self.comment_data[:int(len(self.comment_data) * (1 - test_persentage))]
        
        self.test_code_data = self.code_data[int(len(self.code_data) * (1 - test_persentage)):]
        self.test_comment_data = self.comment_data[int(len(self.comment_data) * (1 - test_persentage)):]
        self.test_raw_comment = self.raw_comment[int(len(self.raw_comment) * (1 - test_persentage)):]

        self.gCodes = self.gCodes[int(len(self.gCodes) * (1 - test_persentage)):]
        self.gComment = self.gComment[int(len(self.gComment) * (1 - test_persentage)):]
                
        


    def MakeDataset(self, train):

        codes = []
        keywords = []

        if train:
            codes = self.train_code_data
            keywords = self.train_comment_data

        else:
            codes = self.test_code_data
            keywords = self.test_comment_data

        train_codes = []
        train_keywords = []

        next_words = np.zeros((len(keywords), self.vocab_size), dtype=np.bool)

        for i, (code, keyword) in enumerate(zip(codes, keywords)):
            train_codes.append(code)
            keyword = set(keyword)
            for index in keyword:
                next_words[i,index] = 1

        return train_codes, next_words




