from utils.data_loader import DataLoader
from utils.hmm_model import HMMBasic
import pickle


if __name__ == '__main__':
    print('*********************************')
    print('***  Name: Truong Nhat Hoang  ***')
    print('***  Student ID: 51703092     ***')
    print('***  HMM POS Tagger           ***')
    print('***  NLP - 504045             ***')
    print('*********************************')
    print()
    print('[Loading data] ...')
    dloader = DataLoader('./data/')
    dloader.load()
    train_word = [j for i in dloader.data_train['label'] for j in i].__len__()
    test_word = [j for i in dloader.data_test['label'] for j in i].__len__()

    print("------[TRAIN SET]------")
    print(f"\t{dloader.data_train.__len__()}: Sentences")
    print(f"\t{train_word}: Words")
    print("------[TEST SET]------")
    print(f"\t{dloader.data_test.__len__()}: Sentences")
    print(f"\t{test_word}: Words")

    print('[TRANING] ...')
    hmm_basic = HMMBasic()
    hmm_basic.train(dloader.data_train)
    score = hmm_basic.eval(dloader.data_test)
    print(f'Accurancy: {score}', end='\n\n')
    print('[SAVING MODEL TO FILE] >>> model/hmm_pos.hmm')
    pickle.dump(hmm_basic, open('model/hmm_pos.hmm', mode='wb'))
