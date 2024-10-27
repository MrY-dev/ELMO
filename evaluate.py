import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,recall_score,confusion_matrix,accuracy_score
from tqdm import tqdm
from classification import Downstreamset, get_seq,DownstreamModel,DownstreamModel2

device = 'cuda'

def evaluate(model,testloader,orig_labels,name):
    for params in model.parameters():
        params.requires_grad =False

    predicted_labels = [] 
    for X,y in tqdm(testloader,total=len(testloader)):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        pred = pred.cpu().detach().numpy()
        predicted_labels.append(pred.argmax() + 1)

    print(name)
    print(f'f1 score :{f1_score(orig_labels,predicted_labels,average="macro")}')
    print(f'recall score: {recall_score(orig_labels,predicted_labels,average="macro")}')
    print(f'accuracy score: {accuracy_score(orig_labels,predicted_labels)}')
    print(f'confusion matrix:\n {confusion_matrix(orig_labels,predicted_labels)}')

def main():
    # trainable lambdas
    trained_lambda_model = torch.load('./trainable_lambdas.pt')
    freezed_lambda_model = torch.load('./untrainable_lambdas.pt')
    trainable_func_model = torch.load('./trainable_function.pt')

    train_data = torch.load('./train_data.pt')
    test_data = torch.load('./test_data.pt')

    inputs = get_seq(test_data,train_data.word2idx)
    orig_labels = test_data.labels
    
    testset = Downstreamset(inputs,orig_labels)
    testloader = DataLoader(testset,shuffle=False)

    print('-------------------------------------------------------------------')
    evaluate(trained_lambda_model,testloader,orig_labels,'trainable lambdas')
    print('-------------------------------------------------------------------')
    evaluate(freezed_lambda_model,testloader,orig_labels,'freezed lambdas')
    print('-------------------------------------------------------------------')
    evaluate(trainable_func_model,testloader,orig_labels,'trainable function of embeddings')

if __name__ == '__main__':
    main()
