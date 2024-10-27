import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datautils import process_data
from datautils import get_seq
import datautils
import numpy as np
from tqdm import tqdm

device = 'cuda'

class bi_lstm(nn.Module):
    def __init__(self,vocab_size,hidden_size,embedding_dim):
        super(bi_lstm,self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.emb_layer = nn.Embedding(vocab_size,embedding_dim)
        self.l1 = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True) 
        self.l2 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,vocab_size)

    def forward(self,x):
        emb = self.emb_layer(x)
        out1,_ = self.l1(emb)
        out2,_ = self.l2(out1)
        final_ind = self.fc(out2)

        return final_ind,out1,out2,emb

class vocab_dataset(Dataset):
    def __init__(self,input,output):
        self.input = pad_sequence(input,batch_first=True)
        self.output = pad_sequence(output,batch_first=True)

    def __len__(self):
        return  len(self.input)

    def __getitem__(self, idx):
        return self.input[idx],self.output[idx]



    
def train(model,dataloader):
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10
    model = model.to(device)
    curr_model = model
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    prev_loss = 1000 
    for epoch in range(num_epochs):
        running_loss = 0
        for X,y in tqdm(dataloader,total=len(dataloader)):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred,_,_,_ = model(X)
            pred = pred.view(pred.shape[0],pred.shape[-1],pred.shape[1])
            y = y.squeeze()
            loss = loss_fn(pred,y)
            running_loss += loss
            loss.backward()
            optimizer.step()
        print(f'running_loss : {running_loss/len(dataloader)} in [{epoch+1}/{num_epochs}]')
        if running_loss/len(dataloader) < prev_loss:
            prev_loss = running_loss/len(dataloader)
            curr_model = model
        # validation step has no batches as padding effects accuracy 
    return curr_model

def main():
    train_data = torch.load('train_data.pt')
    #train validation split
    forward_seqs = get_seq(train_data,train_data.word2idx,reverse=False)
    backward_seqs = get_seq(train_data,train_data.word2idx,reverse=True)
    input = [torch.tensor(seq[:-1],dtype = torch.long) for seq in forward_seqs]
    output = [torch.tensor(seq[1:],dtype = torch.long) for seq in forward_seqs]
    rev_input = [torch.tensor(seq[:-1],dtype = torch.long) for seq in backward_seqs]
    rev_output = [torch.tensor(seq[1:],dtype = torch.long) for seq in backward_seqs]

    vocab_size= len(train_data.word2idx) + 1

    #datasets for forward,backward train and validation splits
    forward_trainset = vocab_dataset(input,output)

    backward_trainset = vocab_dataset(rev_input,rev_output)


    batch_size = 16
    # forward train loader and validation
    forward_loader = DataLoader(forward_trainset,batch_size=batch_size)

    #backward train loader and validation
    backward_loader = DataLoader(backward_trainset,batch_size=batch_size)

    embedding_dim = 200
    # definition and training of forward and backward model
    forward_model = bi_lstm(vocab_size=vocab_size,embedding_dim=embedding_dim,hidden_size=embedding_dim)
    backward_model = bi_lstm(vocab_size=vocab_size,embedding_dim=embedding_dim,hidden_size=embedding_dim)

    print('training forward LM')
    forward_model  = train(forward_model,forward_loader)
    torch.save(forward_model,'forward_model.pt')

    print('training backward LM')
    backward_model   = train(backward_model,backward_loader)
    torch.save(backward_model,'backward_model.pt')

if __name__ == '__main__':
    main()

    
