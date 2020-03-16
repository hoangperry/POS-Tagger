import pickle


if __name__ == '__main__':
    print('*********************************')
    print('***  Name: Truong Nhat Hoang  ***')
    print('***  Student ID: 51703092     ***')
    print('***  HMM POS Tagger           ***')
    print('***  NLP - 504045             ***')
    print('*********************************')

    hmm_model = pickle.load(open('hmm_basic.hmm', 'rb'))
    is_loop = True
    while is_loop:
        input_text = input('Enter a senteces: ')
        output = hmm_model.predict_raw_text(input_text)
        print(output)
        _again = input('Do you want to try again (Enter y/n): ')
        if _again.strip().lower() not in ['y', 'n']:
            _again = input('(Enter Y/n): ')
        if _again.strip().lower() != 'y':
            is_loop = False
