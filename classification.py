import datautils
from ELMO import bi_lstm, get_seq, pad_sequence, process_data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

device = 'cuda'

class DownstreamModel(nn.Module):
    def __init__(self, hidden_dim,embedding_dim,forward_model,backward_model,trainable) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_dim,batch_first=True)
        self.forward_model = forward_model
        self.backward_model = backward_model
        
        for parms in self.forward_model.parameters():
            parms.requires_grad =False

        for parms in self.backward_model.parameters():
            parms.requires_grad =False

        self.l = torch.randn(3,requires_grad=trainable)
        self.fc = nn.Linear(hidden_dim,4)
        
    def forward(self,x):
        l =self.l
        x = x.squeeze()
        _,fe1,fe2,fe3 = self.forward_model(x)
        _,be1,be2,be3 = self.backward_model(torch.flip(x,[0]))
        be1 = torch.flip(be1,dims=[0])
        be2 = torch.flip(be2,dims=[0])
        be3 = torch.flip(be2,dims=[0])

        e1 = torch.cat((fe1,be1),1)
        e2 = torch.cat((fe2,be2),1)
        e3 = torch.cat((fe3,be3),1)

        emb = l[0]*e1 + l[1]*e2 + l[2]*e3
        out,_ = self.lstm(emb)
        final_ind = self.fc(out.mean(dim=0))
        return final_ind

class DownstreamModel2(nn.Module):
    def __init__(self, hidden_dim,embedding_dim,forward_model,backward_model) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_dim,batch_first=True)
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.trainalbe_layer = nn.Linear(3*embedding_dim,embedding_dim)
        
        for parms in self.forward_model.parameters():
            parms.requires_grad =False

        for parms in self.backward_model.parameters():
            parms.requires_grad =False

        self.fc = nn.Linear(hidden_dim,4)
        
    def forward(self,x):
        x = x.squeeze()
        _,fe1,fe2,fe3 = self.forward_model(x)
        _,be1,be2,be3 = self.backward_model(torch.flip(x,[0]))
        be1 = torch.flip(be1,dims=[0])
        be2 = torch.flip(be2,dims=[0])
        be3 = torch.flip(be2,dims=[0])

        e1 = torch.cat((fe1,be1),1)
        e2 = torch.cat((fe2,be2),1)
        e3 = torch.cat((fe3,be3),1)

        emb = self.trainalbe_layer(torch.cat((e1,e2,e3),1))
        out,_ = self.lstm(emb)
        final_ind = self.fc(out.mean(dim=0))
        return final_ind

class Downstreamset(Dataset):
    def __init__(self,inputs,labels):

        inputs = [torch.tensor(input) for input in inputs]
        labels = [torch.tensor(output-1) for output in labels]

        self.input = inputs
        self.output = labels

    def __len__(self):
        return len(self.input)

    def __getitem__(self,idx):
        return self.input[idx],self.output[idx]

def train(model,dataloader,testloader,name):
    num_epochs = 10
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-4)
    prev_acc = 0
    for epochs in range(num_epochs):
        model.train()
        running_loss = 0
        for X,y in tqdm(dataloader,total=len(dataloader),desc='training'):
            X = X.to(device)
            y = y.to(device)
            y = y.squeeze()
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred,y)
            running_loss += loss
            loss.backward() 
            optimizer.step()
        print(f'running loss:{running_loss/len(dataloader)} , in epoch [{epochs+1}/{num_epochs}]')
        model.eval()
        with torch.no_grad():
            acc = 0
            for X,y in tqdm(testloader,total=len(testloader),desc='validation'):
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                acc += (F.softmax(pred,dim=0).argmax() == y)
            acc = acc/len(testloader)
            if  acc > prev_acc:
                print(f'saving the model')
                torch.save(model,name)
                prev_acc = acc
        print(f'val acc:{acc} , in epoch [{epochs+1}/{num_epochs}]')
                
    return model


def main():
    forward_model:bi_lstm = torch.load('forward_model.pt')
    backward_model:bi_lstm = torch.load('backward_model.pt')

    # get test sequences
    train_data = torch.load('train_data.pt')
    test_data = torch.load('test_data.pt')

    test_seqs = datautils.get_seq(test_data,train_data.word2idx)
    train_seqs = datautils.get_seq(train_data,train_data.word2idx)
    
    train_set = Downstreamset(inputs=train_seqs,labels=train_data.labels)
    test_set = Downstreamset(inputs=test_seqs,labels=test_data.labels)
    
    hidden_dim = 200
    embedding_dim = 400
    
    trainloader = DataLoader(train_set,shuffle=True)
    testloader = DataLoader(test_set,shuffle=False)

    model_1 = DownstreamModel(hidden_dim,embedding_dim,forward_model,backward_model,True)
    model_2 = DownstreamModel(hidden_dim,embedding_dim,forward_model,backward_model,False)
    model_3 = DownstreamModel2(hidden_dim,embedding_dim,forward_model,backward_model)

    model_1 = train(model_1,trainloader,testloader,'trainable_lambdas.pt')
    model_2 = train(model_2,trainloader,testloader,'untrainable_lambdas.pt')
    model_3 = train(model_3,trainloader,testloader,'trainable_function.pt')

if __name__ == '__main__':
    main()
