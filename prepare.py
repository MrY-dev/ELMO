import datautils
import torch
import sys

TRAIN_FILE = './ANLP-2/train.csv'
TEST_FILE = './ANLP-2/test.csv'

def main():
    if len(sys.argv) == 1:
        limit = -1
    else:
        limit = int(sys.argv[1])

    print(f'saving train_data and test_data with {limit}')
    train_data = datautils.process_data(TRAIN_FILE,limit=limit,train=True)
    test_data = datautils.process_data(TEST_FILE,train=False)
    torch.save(train_data,'train_data.pt')
    torch.save(test_data,'test_data.pt')
        

if __name__ == '__main__':
    main()

