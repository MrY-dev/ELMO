from nltk.stem.snowball import stopwords
import pandas as pd
import torch
import nltk
import string
from nltk.tokenize import word_tokenize

data_path = './ANLP-2/'


train_file = data_path + 'train.csv'
test_file = data_path + 'test.csv'

class process_data(object):
    def __init__(self,path,train=True,limit=-1):
        print('processing data from ',path)
        df = pd.read_csv(path)
        rows = df['Description']
        labels = df['Class Index']
        if limit == -1:
            rows = rows[::]
            labels = labels[::]
        else:
            rows = rows[:limit]
            labels = labels[:limit]

        print(len(rows),len(labels))
        ps = nltk.stem.PorterStemmer()
        freq = {}
        vocab = []
        sents = []
        #sentence preprocessing

        table = str.translate('',string.punctuation)
        for row in rows:
            row.replace('//',' ').replace("'",'').replace('-',' ')
            row = row.translate(table)
            row = [word.lower() for word in word_tokenize(row) if word.isalpha()]
            #row = [word for word in row if word not in stop_words]
            row = [ps.stem(word) for word in row]
            row = ['<bos>'] + row + ['<eos>']
            if train:
                for word in row:
                    freq[word] = freq.get(word,0) + 1
            sents.append(row)
        self.sents = sents
        self.labels = labels

        # generate vocab and word2idx only for train data and drop words whose frequency is less than 1
        if train:
            for key,val in freq.items():
                if val >= 10:
                    vocab.append(key)
            vocab.append('<unk>')
            vocab = list(set(vocab))
            self.vocab = vocab
            # 0 for pad 
            self.word2idx = {word: idx+1 for idx,word in enumerate(vocab)}
            self.idx2word = {idx+1: word for idx,word in enumerate(vocab)}

    def get_sents(self):
        return self.sents

    def get_labels(self):
        return self.labels

def get_seq(data:process_data,word2idx,reverse=False):
    seqs = []
    for sent in data.get_sents():
        if reverse:
            seqs.append([word2idx.get(word,word2idx['<unk>']) for word in sent[::-1]]) 
        else:
            seqs.append([word2idx.get(word,word2idx['<unk>']) for word in sent]) 
    return seqs


if __name__ == '__main__':
    train_data = process_data(train_file)
    test_data = process_data(test_file,train=False)
    input =[torch.tensor(seq[:-1],dtype = torch.long) for seq in get_seq(train_data,train_data.word2idx)]
    output =[torch.tensor(seq[1:],dtype = torch.long) for seq in get_seq(train_data,train_data.word2idx)]
    print(input[0],output[0])

