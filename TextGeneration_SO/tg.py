




index2word = []
word2index = []



with open("../qnaData/comment_f_Vocab.txt") as fin:
    for i, word in enumerate(fin):
        word = word.rstrip()
        index2word.append(word)

id=[]
newS=[]

stopWord = [',',"'",'"',':','<','>','-']
with open("ref.txt") as f:
    for line in f:
        line = line.rstrip()
        tokens = line.split('\t')
        sentence = tokens[1]

        sentence = sentence.replace('?', ' ?')
        for w in stopWord:
            if w in sentence:
                sentence.replace(w,' ')

        words = sentence.split(' ')
        newSentence = []
        for word in words:
            word = word.lower()
            if word == '':
                continue
            if word in index2word:
                newSentence.append(word)
            else:
                newSentence.append('UNK')

        newS.append(' '.join(newSentence))
        id.append(tokens[0])
with open("ref_UNK.txt",'w') as f:
    for i, s in zip(id,newS):
        f.write(i+'\t'+s+'\n')

