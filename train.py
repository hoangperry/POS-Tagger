from data_loader import DataLoader
from hmm_model import HMMBasic
import pickle


if __name__ == '__main__':
    dloader = DataLoader('./data/')
    dloader.load()
    hmm_basic = HMMBasic()
    hmm_basic.train(dloader.data_train)
    score = hmm_basic.eval(dloader.data_test)
    print(score)
    pickle.dump(hmm_basic, open('hmm_basic.hmm', mode='wb'))
