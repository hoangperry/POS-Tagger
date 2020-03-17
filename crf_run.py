import pickle
from crf_train import get_features

if __name__ == '__main__':
    print('*********************************')
    print('***  Name: Truong Nhat Hoang  ***')
    print('***  Student ID: 51703092     ***')
    print('***  CRF POS Tagger           ***')
    print('***  NLP - 504045             ***')
    print('*********************************')

    crf_model = pickle.load(open('crf_pos_model.crf', 'rb'))
    is_loop = True
    while is_loop:
        input_text = input('Enter a senteces: ')
        input_text = input_text.split(' ')
        input_text_features = get_features([(i, '') for i in input_text])
        output = crf_model.predict([input_text_features])[0]
        for i in range(output.__len__()):
            print('({}|{}) '.format(input_text[i], output[i]), end='')
        print('\n')
        _again = input('Do you want to try again (Enter y/n): ')
        if _again.strip().lower() not in ['y', 'n']:
            _again = input('(Enter Y/n): ')
        if _again.strip().lower() != 'y':
            is_loop = False
